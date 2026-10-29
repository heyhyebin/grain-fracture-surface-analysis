"""
지식 카드 검색 모듈 (/report 보고서용)

흐름
  1) knowledge/fractography_cards.jsonl 의 카드를 읽는다.
     카드 한 장 = 문헌에서 요약한 사실 하나 + 출처 (ASM Handbook Vol.12, 리뷰 논문 등)
  2) CNN 예측 유형, 재질, SEM 관찰 가능 여부, 배율 필요 여부로 사용할 카드를 거른다.
  3) 보고서 항목(slot)마다 이전 분석의 관찰 문장과 가장 가까운 카드를 몇 장씩 고른다.
  4) llm_service.generate_report() 가 고른 카드를 프롬프트에 넣고,
     LLM 이 인용한 카드 ID 를 응답의 knowledge_refs 로 프론트에 내려준다.

설계 메모
  - 카드가 100여 장이라 벡터 DB 없이 numpy 행렬 하나로 충분하다.
  - 임베딩은 rag_service 와 같은 bge-m3 를 쓰고 knowledge/card_embeddings.npy 에 캐시한다.
    카드 파일이 바뀌면(sha1 이 달라지면) 다음 호출에서 다시 만든다.
  - 임베딩을 쓸 수 없으면(bge-m3 미설치, Ollama 오류) 카드 파일 순서대로 고른다.
    카드 선택이 실패해도 보고서 생성은 계속된다.
  - 데이터셋 SEM 이미지에는 배율 정보가 없으므로 requires_scale 카드(딤플 크기,
    줄무늬 간격 등)는 쓰지 않는다. sem_observable=false 카드(비치 마크 등 육안 특징)는
    SEM 사진에서 관찰될 수 없으므로 역시 쓰지 않는다.
"""

import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional

import numpy as np

from imaging_info import has_scale


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
CARDS_PATH = os.path.join(KNOWLEDGE_DIR, "fractography_cards.jsonl")
EMB_PATH = os.path.join(KNOWLEDGE_DIR, "card_embeddings.npy")
EMB_META_PATH = os.path.join(KNOWLEDGE_DIR, "card_embeddings.json")

CNN_TYPES = ("Cleavage", "Ductile", "Fatigue", "Intergranular")

UNKNOWN_MATERIALS = ("", "unknown", None)

# General 카드 중 특정 유형에만 관련된 카드. 여기에 없는 General 카드는 네 유형 모두에 쓴다.
RELATED_TYPES = {
    "DIF-001": ("Cleavage", "Intergranular"),
    "DIF-002": ("Fatigue",),
    "DIF-004": ("Cleavage", "Intergranular"),
    "MG-002": ("Fatigue",),
}

# 보고서 대상 시편과 직접 관계없는 카드 (선행 연구 정확도, 조직 차원의 예방 절차)
EXCLUDED_FROM_REPORT = {"GEN-006", "PRV-001"}

# (slot, 사용할 section, 최대 장수, 검색 질의 종류)
SLOTS = (
    ("feature", ("feature",), 2, "obs"),
    ("differential", ("differential",), 1, "obs"),
    ("model_basis", ("model_basis",), 2, "cam"),
    ("cause", ("cause", "mechanism"), 2, "obs"),
    ("caution", ("caution",), 1, "obs"),
    ("next_step", ("next_step",), 1, "obs"),
)

# 재질을 입력했을 때 재질 전용 카드를 추가로 넣는 장수
MATERIAL_SLOT_SIZE = 1

# 검색 결과와 관계없이 항상 넣는 카드 (Grad-CAM++ 강조 영역 ≠ 실제 결함 위치)
PINNED = {"model_basis": ("XAI-002",)}

# 사용자가 SEM 촬영 정보를 입력했을 때 우선 넣는 카드 (입력 항목, 값) → 카드 ID 순서.
# 예측 유형·재질에 맞지 않는 카드는 _eligible 에서 걸러진다.
IMAGING_PINS = {
    ("location", "origin"): ("LOC-002", "LOC-005", "LOC-001", "LOC-006", "LOC-004", "FAT-012"),
    ("location", "propagation"): ("IMG-001", "LOC-010", "FAT-003", "LOC-008", "IMG-003", "LOC-007"),
    ("location", "final"): ("LOC-003", "LOC-010", "DIF-005", "LOC-009", "FAT-008", "LOC-012"),
    ("location", "overview"): ("MAG-001", "IMG-001", "LOC-011", "IMG-004", "SEM-003"),
    ("detector", "BSE"): ("SEM-001", "DET-001", "DET-002"),
    ("detector", "SE"): ("SEM-002",),
    ("cleaned", "no"): ("DMG-007", "DMG-001", "STR-003"),
    ("cleaned", "yes"): ("CLN-001",),
    ("coated", "yes"): ("COAT-001",),
    ("coated", "no"): ("SEM-005",),
    ("scale", True): ("MAG-001", "MAG-004", "STR-001", "SCL-001", "IMG-001", "MAG-003",
                      "SEM-003", "FAT-014", "DUC-011"),
}

# 입력 항목별 최대 장수와 전체 최대 장수 (프롬프트 길이 제한).
# 위치와 배율을 함께 입력해도 한쪽이 다른 쪽 카드를 모두 밀어내지 않도록 항목별로 나눈다.
IMAGING_KEY_LIMITS = {"location": 3, "scale": 2, "detector": 2, "cleaned": 1, "coated": 1}
IMAGING_SLOT_SIZE = 6

SECTION_LABELS = {
    "feature": "특징",
    "mechanism": "메커니즘",
    "cause": "원인",
    "material": "재질",
    "next_step": "확인 방법",
    "caution": "주의",
    "differential": "유형 감별",
    "model_basis": "Grad-CAM++ 해석",
}


# =========================================================
# 카드 저장소
# =========================================================

def _card_document(card: Dict) -> str:
    """임베딩할 텍스트: 본문 + 영문 키워드 (bge-m3 는 한/영 혼용 검색 가능)."""
    return card["text"] + "\n" + ", ".join(card.get("keywords", []))


class CardStore:

    def __init__(self, path: Optional[str] = None):
        path = path or CARDS_PATH
        self.path = path
        self.cards: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self._digest = ""
        self._embedding_tried = False

        with open(path, "rb") as f:
            raw = f.read()

        self._digest = hashlib.sha1(raw).hexdigest()

        for line in raw.decode("utf-8").splitlines():
            line = line.strip()
            if line:
                self.cards.append(json.loads(line))

    # ---------------------------------------------------
    # 임베딩 (지연 생성 + 캐시)
    # ---------------------------------------------------

    def _ensure_embeddings(self) -> None:
        if self.embeddings is not None or self._embedding_tried:
            return

        self._embedding_tried = True

        try:
            from rag_service import EMBED_MODEL, embed_texts

            cached = self._load_cache(EMBED_MODEL)

            if cached is not None:
                self.embeddings = cached
                return

            print(f"[지식 카드] 카드 {len(self.cards)}장 임베딩 생성 중 ({EMBED_MODEL})")

            emb = embed_texts([_card_document(c) for c in self.cards])

            os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)
            np.save(EMB_PATH, emb)

            with open(EMB_META_PATH, "w", encoding="utf-8") as f:
                json.dump(
                    {"embed_model": EMBED_MODEL, "cards_sha1": self._digest,
                     "count": len(self.cards)},
                    f,
                )

            self.embeddings = emb

        except Exception as e:
            print(f"[지식 카드] 임베딩 사용 불가 → 카드 순서대로 선택: {e}")
            self.embeddings = None

    def _load_cache(self, embed_model: str) -> Optional[np.ndarray]:
        if not (os.path.exists(EMB_PATH) and os.path.exists(EMB_META_PATH)):
            return None

        with open(EMB_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if meta.get("embed_model") != embed_model or meta.get("cards_sha1") != self._digest:
            return None

        emb = np.load(EMB_PATH, allow_pickle=False)

        return emb if emb.shape[0] == len(self.cards) else None

    def embed_query(self, query: str) -> Optional[np.ndarray]:
        """질의 벡터. 임베딩을 쓸 수 없으면 None."""
        self._ensure_embeddings()

        if self.embeddings is None or not query.strip():
            return None

        try:
            from rag_service import embed_texts

            q = embed_texts([query])[0]

            return q if q.shape[0] == self.embeddings.shape[1] else None

        except Exception as e:
            print(f"[지식 카드] 질의 임베딩 실패 → 카드 순서대로 선택: {e}")
            return None

    def rank(self, idxs: List[int], q: Optional[np.ndarray]) -> List[int]:
        """질의와 가까운 순으로 정렬. 질의 벡터가 없으면 파일 순서 그대로."""
        if q is None or not idxs:
            return idxs

        scores = self.embeddings[idxs] @ q

        return [idxs[i] for i in np.argsort(-scores, kind="stable")]


_store: Optional[CardStore] = None


def get_store() -> Optional[CardStore]:
    global _store

    if _store is None:
        try:
            _store = CardStore()
            print(f"[지식 카드] {len(_store.cards)}장 로드")
        except Exception as e:
            print(f"[지식 카드] 카드 파일 로드 실패: {e}")
            return None

    return _store


# =========================================================
# 카드 선택
# =========================================================

def _eligible(card: Dict, prediction_en: str, material: str,
              banned_words: Iterable[str], allow_scale: bool = False) -> bool:

    if card["id"] in EXCLUDED_FROM_REPORT:
        return False

    if card["type"] == "General":
        if prediction_en not in RELATED_TYPES.get(card["id"], CNN_TYPES):
            return False
    elif card["type"] != prediction_en:
        return False

    if card.get("material", "any") not in ("any", material):
        return False

    # 배율·스케일 바를 입력하지 않았으면 크기·간격 카드 제외 / SEM 사진에서 볼 수 없는 육안 특징 제외
    if (card.get("requires_scale") and not allow_scale) or card.get("sem_observable") is False:
        return False

    if any(w in card["text"] for w in banned_words):
        return False

    return True


def _imaging_pin_groups(imaging: Optional[Dict]) -> List[tuple]:
    """입력된 촬영 정보별 (항목명, 후보 카드 ID 목록). 입력하지 않은 항목은 빠진다."""
    imaging = imaging or {}
    keys = [(k, imaging.get(k)) for k in ("location", "scale", "detector", "cleaned", "coated")]
    keys[1] = ("scale", has_scale(imaging))

    return [(key[0], IMAGING_PINS[key]) for key in keys if key in IMAGING_PINS]


def select_cards(prediction_en: str, material: str = "", obs_query: str = "",
                 cam_query: str = "", banned_words: Iterable[str] = (),
                 imaging: Optional[Dict] = None) -> List[Dict]:
    """
    보고서 항목별로 카드를 고른다. 반환: 카드 dict 목록 (각 카드에 "slot" 추가).
    imaging: imaging_info.normalize() 결과. 입력된 촬영 정보에 맞는 카드를 먼저 넣고,
             배율·스케일 바가 있으면 크기·간격 카드도 허용한다. 없거나 모두 모름이면 기존과 같다.
    카드 파일이 없거나 오류가 나면 빈 리스트.
    """
    store = get_store()

    if store is None or prediction_en not in CNN_TYPES:
        return []

    material = "" if material in UNKNOWN_MATERIALS else material
    banned_words = tuple(banned_words)
    allow_scale = has_scale(imaging)

    eligible = [
        i for i, c in enumerate(store.cards)
        if _eligible(c, prediction_en, material, banned_words, allow_scale)
    ]

    queries = {
        "obs": store.embed_query(obs_query),
        "cam": store.embed_query(cam_query) if cam_query else None,
    }

    chosen: List[int] = []
    slots: Dict[int, str] = {}

    def take(slot: str, candidates: List[int], k: int, query_kind: str) -> None:
        candidates = [i for i in candidates if i not in slots]

        pinned = [i for i in candidates if store.cards[i]["id"] in PINNED.get(slot, ())]

        # 같은 항목 안에서는 예측 유형 전용 카드를 일반(General) 카드보다 먼저 쓴다.
        ranked = store.rank([i for i in candidates if i not in pinned], queries[query_kind])
        ranked.sort(key=lambda i: store.cards[i]["type"] == "General")

        for i in (pinned + ranked)[:k]:
            chosen.append(i)
            slots[i] = slot

    # 촬영 정보 카드: 입력 항목마다 IMAGING_KEY_LIMITS 장까지, 전체 IMAGING_SLOT_SIZE 장까지
    id_to_idx = {store.cards[i]["id"]: i for i in eligible}
    for key, cids in _imaging_pin_groups(imaging):
        taken = 0
        for cid in cids:
            if taken >= IMAGING_KEY_LIMITS.get(key, 1) or len(chosen) >= IMAGING_SLOT_SIZE:
                break
            i = id_to_idx.get(cid)
            if i is not None and i not in slots:
                chosen.append(i)
                slots[i] = "imaging"
                taken += 1

    for slot, sections, k, query_kind in SLOTS:
        take(slot, [i for i in eligible if store.cards[i]["section"] in sections], k, query_kind)

    if material:
        take("material",
             [i for i in eligible if store.cards[i].get("material") == material],
             MATERIAL_SLOT_SIZE, "obs")

    return [dict(store.cards[i], slot=slots[i]) for i in chosen]


# =========================================================
# 프롬프트 / 응답 변환
# =========================================================

def format_cards_for_prompt(cards: List[Dict]) -> str:
    return "\n".join(
        f"[{c['id']}] ({SECTION_LABELS.get(c['section'], c['section'])}) {c['text']}"
        for c in cards
    )


def parse_used_cards(value, cards: List[Dict]) -> List[str]:
    """LLM 이 적은 카드 ID 중 실제로 제공한 카드만 남긴다."""
    valid = {c["id"] for c in cards}

    if isinstance(value, str):
        value = value.replace(",", " ").split()

    if not isinstance(value, list):
        return []

    used = []

    for item in value:
        cid = str(item).strip().strip("[]").upper()
        if cid in valid and cid not in used:
            used.append(cid)

    return used


def _citation(source: Dict) -> Dict:
    venue = source.get("book") or source.get("journal") or ""

    parts = [source.get("authors", ""), source.get("title", ""), venue,
             str(source.get("year", ""))]

    text = ", ".join(p for p in parts if p)

    if source.get("locator"):
        text += f" — {source['locator']}"

    url = source.get("url")

    if not url and source.get("doi"):
        url = f"https://doi.org/{source['doi']}"

    return {"citation": text, "url": url}


def public_cards(cards: List[Dict], used_ids: List[str]) -> List[Dict]:
    """프론트에 내려줄 카드 정보. 보고서에 인용된 카드가 앞에 온다."""
    used = set(used_ids)

    items = [
        {
            "id": c["id"],
            "section": SECTION_LABELS.get(c["section"], c["section"]),
            "text": c["text"],
            "cited": c["id"] in used,
            "sources": [_citation(s) for s in c.get("sources", [])],
        }
        for c in cards
    ]

    return sorted(items, key=lambda x: not x["cited"])


if __name__ == "__main__":
    # 카드 임베딩을 미리 만들어 두려면: python knowledge_cards.py
    store = get_store()

    if store is not None:
        store._ensure_embeddings()
        print("임베딩:", "준비 완료" if store.embeddings is not None else "사용 불가 (카드 순서대로 선택)")