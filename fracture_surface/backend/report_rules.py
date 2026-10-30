"""
입력 조건 → 보고서 필수 반영 사항 (규칙 기반)

LLM 에게 "카드를 보고 알아서 반영하라"고 맡기면 작은 모델은 입력 조건을 무시한다.
그래서 조건 대조(촬영 위치·배율·재질·2위 확률 × CNN 예측)는 파이썬에서 먼저 판단하고,
LLM 은 여기서 만든 결론 문장을 지정된 필드에 풀어 쓰기만 한다.

build_context() 반환값
- must_lines     : [(필드, 지시문)]  보고서 프롬프트의 [필수 반영 사항]
- obs_hints      : [지시문]          분석(관찰) 프롬프트에 넣는 관찰 지시
- cause_candidates: [후보]           분석 프롬프트 expected_cause 후보 (조건에 맞는 것이 맨 앞)
- low_mag        : bool              저배율 → 분류 신뢰도 낮음
- reliability_note: str             신뢰도 박스에 덧붙일 문장
- extra_steps    : [{title, detail}] 확인 사항에 추가할 항목
- background_cards: [카드 ID]        배경 정보 때문에 우선 넣을 지식 카드
"""

from typing import Dict, List, Optional, Tuple

import background_info
import imaging_info

KO = {
    "Cleavage": "취성 파괴",
    "Ductile": "연성 파괴",
    "Fatigue": "피로 파괴",
    "Intergranular": "입계 파괴",
}
EN = {v: k for k, v in KO.items()}

# 스케일 바가 이 값(µm) 이상이거나 배율이 이 값 이하이면 저배율로 본다.
LOW_MAG_SCALE_UM = 100.0
LOW_MAG_MAGNIFICATION = 200.0

# 2위 유형을 "혼합 가능성"으로 언급할 최소 확률(%)
SECOND_CLASS_MIN = 20.0

# 벽개가 드문 면심입방(FCC) 계열 재질
FCC_MATERIALS = {"aluminum": "알루미늄", "copper": "구리", "nickel_alloy": "니켈 합금",
                 "hea": "고엔트로피 합금(HEA)"}

# 배경 정보 → (예측 유형별) 우선 카드. 카드의 유형·재질 필터와 관계없이 넣는다
# (재질 전용 카드도 '다른 재질 사례'로 읽히도록 카드 본문에 재질이 적혀 있다).
BACKGROUND_CARDS = {
    ("environment", "hydrogen"): {
        "Ductile": ["HYD-004", "DIF-005"],
        "Cleavage": ["CLV-011", "HYD-002", "HYD-001"],
        "Intergranular": ["IGF-003", "CLV-011", "HYD-002"],
        "Fatigue": ["DIF-002"],
    },
    ("environment", "chloride"): {
        "Intergranular": ["IGF-015", "IGF-003"],
        "Ductile": ["DIF-005"],
        "Fatigue": ["DMG-001", "STR-003"],
        "Cleavage": ["CLV-012"],
    },
    ("environment", "chemical"): {
        "Intergranular": ["IGF-015", "IGF-003"],
        "Ductile": ["DIF-005"],
        "Fatigue": ["DMG-001"],
    },
    ("temperature", "high"): {
        "Ductile": ["DUC-016", "DIF-005"],
        "Intergranular": ["IGF-011"],
        "Fatigue": ["FAT-015"],
    },
    ("temperature", "low"): {
        "Cleavage": ["CLV-010", "CLV-008"],
    },
    ("load", "impact"): {
        "Cleavage": ["CLV-010"],
        "Ductile": ["SCL-001"],
    },
    ("load", "cyclic"): {
        "Ductile": ["LOC-003", "DIF-005"],
        "Cleavage": ["LOC-003"],
        "Intergranular": ["LOC-003"],
    },
    ("load", "static"): {
        "Fatigue": ["DIF-002", "IMG-002"],
    },
    ("process", "heat_treated"): {
        "Intergranular": ["IGF-005", "IGF-015"],
    },
    ("process", "plated"): {
        "Intergranular": ["IGF-004"],
        "Cleavage": ["CLV-011"],
    },
    ("process", "welded"): {
        "Fatigue": ["FAT-012"],
        "Intergranular": ["IGF-012"],
    },
}

# 기본 원인 후보 (조건 규칙이 앞에 끼워 넣는다)
BASE_CAUSES = {
    "Ductile": [
        "과도한 단일 하중(과부하)에 의한 연성 파단 가능성",
        "선행 균열이 진행된 뒤 남은 단면이 과부하로 파단되었을 가능성",
        "국부적인 변형 집중에 따른 연성 파단 가능성",
    ],
    "Cleavage": [
        "충격성 하중 또는 저온 조건에서의 취성 파단 가능성",
        "응력 집중부에서 시작된 빠른 균열 진행 가능성",
        "재료 취화(열처리·환경)에 따른 저에너지 파단 가능성",
    ],
    "Fatigue": [
        "반복 하중에 따른 점진적 균열 성장 가능성",
        "응력 집중부(노치·결함)에서 시작된 피로 균열 가능성",
        "진동 등 운전 중 반복 응력의 영향 가능성",
    ],
    "Intergranular": [
        "결정립계 취화(편석·석출)에 따른 입계 분리 가능성",
        "수소 또는 부식 환경에 의한 환경 보조 입계 균열 가능성",
        "고온 노출에 따른 입계 손상 가능성",
    ],
}

# 스케일이 있을 때 유형별로 상대 크기를 비교할 대상
SCALE_TARGET = {
    "Ductile": "딤플",
    "Cleavage": "평탄한 벽개면(파셋)",
    "Fatigue": "줄무늬 간격",
    "Intergranular": "결정립 면",
}


def _to_percent(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v * 100 if v <= 1.0 else v
    s = str(value).replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _second_class(prediction_en: str, probabilities) -> Optional[Tuple[str, float]]:
    """probabilities: {한글/영문 이름: '65.3%' | 0.653 | '-'} → (한글 이름, %)"""
    if not probabilities:
        return None
    best = None
    for name, value in dict(probabilities).items():
        en = name if name in KO else EN.get(name)
        if not en or en == prediction_en:
            continue
        p = _to_percent(value)
        if p is None:
            continue
        if best is None or p > best[1]:
            best = (KO[en], p)
    return best


def is_low_mag(imaging: Optional[Dict]) -> bool:
    info = imaging or {}
    if info.get("location") == "overview":
        return True
    scale = info.get("scale_bar_um")
    if scale and float(scale) >= LOW_MAG_SCALE_UM:
        return True
    mag = info.get("magnification")
    if mag and float(mag) <= LOW_MAG_MAGNIFICATION:
        return True
    return False


def build_context(prediction_en: str, material: str = "",
                  imaging: Optional[Dict] = None, probabilities=None,
                  background: Optional[Dict] = None) -> Dict:
    pred_ko = KO.get(prediction_en, prediction_en)
    imaging = imaging_info.normalize(imaging)
    background = background_info.normalize(background)
    location = imaging.get("location")
    scale = imaging.get("scale_bar_um")
    low_mag = is_low_mag(imaging)

    must: List[Tuple[str, str]] = []
    obs: List[str] = []
    causes: List[str] = list(BASE_CAUSES.get(prediction_en, []))
    notes: List[str] = []
    steps: List[Dict] = []

    def first_cause(text: str) -> None:
        if text in causes:
            causes.remove(text)
        causes.insert(0, text)

    # ── 배율·스케일 ─────────────────────────────────────
    if low_mag:
        obs.append(
            "저배율 이미지다. 딤플·줄무늬 같은 미세 형상을 하나하나 보았다고 쓰지 말고, "
            "영역 단위의 거칠기·평탄도·경계·영역 간 차이만 기록한다."
        )
        must.append(("basis",
                     "이 이미지는 저배율이며 CNN은 고배율 SEM 이미지로 학습되었으므로 "
                     "분류 결과는 참고용이라는 점을 밝힌다."))
        must.append(("characteristics",
                     "저배율이라 줄무늬·딤플 같은 미세 형상은 이 이미지로 확인할 수 없다는 점을 밝히고, "
                     "보이는 영역 단위 차이(거친 영역과 평탄한 영역 등)만 서술한다."))
        notes.append("저배율 이미지는 학습 데이터(고배율 SEM)와 달라 분류 신뢰도가 낮습니다. "
                     "같은 영역을 고배율로 촬영해 다시 분석하세요.")
        steps.append({"title": "고배율 재촬영",
                      "detail": "저배율에서 영역을 나눈 뒤 각 영역을 수백~수천 배로 다시 촬영해 개별 분석하세요."})
    elif scale:
        target = SCALE_TARGET.get(prediction_en, "주요 형상")
        obs.append(
            f"스케일 바는 {imaging_info._fmt(scale)} µm 이다. 주요 형상의 크기를 스케일 바 길이와 비교해 "
            "'스케일 바보다 훨씬 작다 / 비슷하다 / 크다'처럼 상대적으로 쓴다. 정확한 수치는 쓰지 않는다."
        )
        must.append(("characteristics",
                     f"스케일 바({imaging_info._fmt(scale)} µm)와 비교한 {target}의 상대 크기"
                     "(훨씬 작다/비슷하다/크다)를 1문장 쓴다. 정확한 수치는 쓰지 않는다."))
        if prediction_en == "Ductile":
            must.append(("cause",
                         "딤플이 스케일 바에 비해 매우 작고 얕아 보이면 국부 소성 변형이 제한적이었을 가능성을, "
                         "크고 깊어 보이면 소성 변형이 컸을 가능성을 관찰과 연결해 쓴다."))

    # ── 촬영 위치 × 예측 ────────────────────────────────
    if location == "final":
        if prediction_en == "Ductile":
            must.append(("cause",
                         "최종 파단부의 딤플은 전체 파손 메커니즘을 대표하지 않으므로 단순 과부하로 단정하지 않는다. "
                         "선행 균열(피로·환경 취화 등)이 진행된 뒤 남은 단면이 과부하로 끊어졌을 가능성을 함께 쓴다."))
            first_cause("선행 균열이 진행된 뒤 남은 단면이 과부하로 파단되었을 가능성")
        elif prediction_en in ("Cleavage", "Intergranular"):
            must.append(("cause",
                         f"최종 파단부가 {pred_ko} 형태라는 것은 마지막 파단이 낮은 에너지로 일어났음을 뜻할 수 있다. "
                         "재료 인성 저하·저온·환경 취화 가능성과, 시작부는 다른 기구였을 가능성을 함께 쓴다."))
        elif prediction_en == "Fatigue":
            must.append(("basis",
                         "최종 파단부에서 피로로 분류된 것은 드문 결과이므로 촬영 위치나 분류를 다시 확인할 필요가 있다고 쓴다."))
            notes.append("최종 파단부 사진이 피로로 분류되었습니다. 촬영 위치가 맞는지, 오분류인지 확인하세요.")
        steps.append({"title": "시작점·전파 영역 확인",
                      "detail": "최종 파단부만으로는 전체 메커니즘을 알 수 없습니다. 저배율로 시작점과 비치 마크를 찾아 해당 영역을 따로 분석하세요."})

    elif location == "origin":
        if prediction_en == "Ductile":
            must.append(("cause",
                         "균열 시작점 부근인데 딤플이 주로 보이면, 과부하로 시작된 파단이거나 실제 시작점이 다른 곳일 가능성이 있다고 쓴다."))
            steps.append({"title": "실제 시작점 재확인",
                          "detail": "시작점 부근에서 딤플만 보이면 촬영 위치가 시작점이 아닐 수 있습니다. 저배율로 방사형 무늬나 비치 마크의 수렴점을 다시 찾으세요."})
        elif prediction_en == "Fatigue":
            must.append(("cause", "시작점 부근의 결함(개재물·가공 흔적·부식 피트·노치)이 피로 균열의 시작에 관여했을 가능성을 쓴다."))
            first_cause("응력 집중부(노치·결함)에서 시작된 피로 균열 가능성")
        elif prediction_en in ("Cleavage", "Intergranular"):
            must.append(("cause",
                         f"시작점이 {pred_ko} 형태이면 재료 취화나 수소·부식 같은 환경 요인이 균열 시작에 관여했을 가능성을 우선 고려한다고 쓴다."))
            if prediction_en == "Intergranular":
                first_cause("수소 또는 부식 환경에 의한 환경 보조 입계 균열 가능성")
        steps.append({"title": "시작점 주변 정밀 관찰",
                      "detail": "시작점 주변의 개재물·기공·가공 흔적·부식 생성물을 고배율과 EDS 성분 분석으로 확인하세요."})

    elif location == "propagation":
        if prediction_en == "Fatigue":
            must.append(("characteristics", "전파 영역에서 관찰되는 줄무늬·비치 마크 관련 형상이 있으면 그 방향과 규칙성을 쓴다."))
        else:
            must.append(("cause",
                         f"균열 전파 영역이 {pred_ko} 형태이면 반복 하중보다 단조 하중이나 환경 보조 균열 성장(응력부식·수소 취화) "
                         "가능성을 함께 고려한다고 쓴다."))

    # ── 검출기 / 세척 ───────────────────────────────────
    if imaging.get("detector") == "BSE":
        obs.append("후방산란전자(BSE) 영상이므로 밝기 차이를 요철이 아니라 원소(원자번호) 차이로 해석할 수 있다.")
        must.append(("characteristics", "BSE 영상의 밝기 차이는 표면 요철이 아니라 원소 차이일 수 있다는 점을 반영한다."))
    if imaging.get("cleaned") == "no":
        must.append(("characteristics",
                     "세척하지 않은 파면이므로 특징 없는 매끈한 영역은 부식 생성물·오염일 수 있으며, 이를 취성의 근거로 쓰지 않는다."))

    # ── 재질 × 예측 ─────────────────────────────────────
    if prediction_en == "Cleavage" and material in FCC_MATERIALS:
        name = FCC_MATERIALS[material]
        must.append(("basis",
                     f"{name}은 대부분 면심입방(FCC) 구조라 일반적으로 벽개 파괴가 드물다. 준벽개·입계 파괴, "
                     "취성 2상 입자의 파괴, 또는 오분류일 가능성을 밝힌다."))
        notes.append(f"{name}은 벽개 파괴가 드문 재질이라 분류 결과와 맞지 않을 수 있습니다.")
        steps.append({"title": "재질-분류 불일치 확인",
                      "detail": "FCC 금속에서 벽개로 분류되면 준벽개, 입계 파괴, 취성 2상 입자 파괴 가능성을 함께 확인하세요."})

    # ── 사전 배경 정보 × 예측 ───────────────────────────
    bg_must, bg_causes, bg_notes, bg_steps = _background_rules(prediction_en, pred_ko, material, background)
    must += bg_must
    for c in reversed(bg_causes):
        first_cause(c)
    notes += bg_notes
    steps = bg_steps + steps

    # ── 2위 유형 (혼합 가능성) ──────────────────────────
    second = _second_class(prediction_en, probabilities)
    if second and second[1] >= SECOND_CLASS_MIN:
        must.append(("basis",
                     f"2위 유형인 {second[0]}({second[1]:.1f}%)의 특징이 일부 영역에 섞였을 가능성(혼합 파괴 모드)을 1문장 쓴다."))

    # ── 연성 기본 주의 ──────────────────────────────────
    # 최종 파단부 규칙이 이미 같은 내용을 요구하는 경우만 빼고, 연성 보고서에는 항상 붙인다.
    if prediction_en == "Ductile" and location != "final" and background.get("load") != "cyclic":
        must.append(("cause", "딤플만으로 상온 과부하라고 단정할 수 없다는 점(크리프·환경 취화 후 최종 파단 등)을 1문장 덧붙인다."))

    return {
        "must_lines": must,
        "obs_hints": obs,
        "cause_candidates": causes,
        "low_mag": low_mag,
        "reliability_note": " ".join(notes),
        "extra_steps": steps,
        "background_cards": _background_cards(prediction_en, background),
    }


def _background_cards(prediction_en: str, background: Dict) -> List[str]:
    out: List[str] = []
    pairs = [("environment", e) for e in background.get("environment", [])]
    pairs += [("process", p) for p in background.get("process", [])]
    pairs += [(k, background.get(k)) for k in ("load", "temperature") if background.get(k)]
    for key in pairs:
        for cid in BACKGROUND_CARDS.get(key, {}).get(prediction_en, []):
            if cid not in out:
                out.append(cid)
    return out


def _background_rules(pred_en: str, pred_ko: str, material: str, bg: Dict):
    """배경 정보와 CNN 예측을 대조해 (필수 반영, 원인 후보(앞에 올 순서), 신뢰도 메모, 확인 사항)을 만든다."""
    must: List[Tuple[str, str]] = []
    causes: List[str] = []
    notes: List[str] = []
    steps: List[Dict] = []

    env = set(bg.get("environment") or [])
    load = bg.get("load")
    temp = bg.get("temperature")
    proc = set(bg.get("process") or [])
    manner = bg.get("failure_manner")

    if background_info.is_empty(bg):
        if pred_en == "Intergranular":
            must.append(("cause", "사용 환경·열 이력 정보가 없어 수소 취화·응력부식·템퍼 취화·크리프를 구분할 수 없다는 점을 밝힌다."))
            steps.append({"title": "사용 환경·열 이력 확인",
                          "detail": "입계 파괴는 수소·부식 환경, 열처리(템퍼 취화·예민화), 고온 노출에 따라 원인이 달라집니다. 이 정보를 확인해 다시 분석하세요."})
        return must, causes, notes, steps

    hydrogen = "hydrogen" in env or "plated" in proc
    corrosive = bool(env & {"chloride", "chemical"})

    # ── 연성 ──
    if pred_en == "Ductile":
        if load == "cyclic":
            must.append(("cause", "반복 하중 조건이므로 이 딤플 영역은 피로 균열이 자란 뒤 남은 단면의 최종 파단부일 가능성을 우선 쓴다."))
            causes.append("선행 균열이 진행된 뒤 남은 단면이 과부하로 파단되었을 가능성")
            steps.append({"title": "비치 마크·줄무늬 확인",
                          "detail": "반복 하중 부품입니다. 저배율로 비치 마크를 찾고, 시작부 쪽을 고배율로 촬영해 피로 줄무늬를 확인하세요."})
        if hydrogen:
            must.append(("cause", "수소 노출 조건에서도 딤플(연성) 파면이 보이므로 이 영역에서는 수소 취화의 영향이 제한적이었을 가능성을 쓴다. "
                                  "다만 수소 농도가 높은 표면·가장자리에는 취성 영역이 따로 있을 수 있다고 덧붙인다."))
            steps.append({"title": "가장자리(표면 근처) 촬영",
                          "detail": "수소는 표면에서 들어오므로 가장자리에 입계·준벽개 같은 취성 영역이 있는지 따로 촬영해 비교하세요."})
        if temp == "high" and load in ("sustained", None, ""):
            must.append(("cause", "고온 조건이므로 딤플처럼 보이는 공극이 크리프 캐비티일 가능성을 함께 쓴다."))
        if temp == "low":
            must.append(("cause", "저온에서도 연성 파면이 나타났다면 재료가 저온 인성을 유지했을 가능성(FCC 계열 등)을 쓴다."))
        if load == "impact":
            must.append(("characteristics", "충격 하중의 딤플은 정적 인장보다 작고 불규칙하며 전단 영향이 클 수 있다는 점과 관찰을 비교한다."))
        if corrosive and load != "cyclic":
            must.append(("cause", "부식 환경이므로 응력부식 균열이 진행된 뒤 남은 단면이 연성 파단되었을 가능성을 함께 쓴다."))

    # ── 취성(벽개) ──
    elif pred_en == "Cleavage":
        if temp == "low" or load == "impact":
            causes.append("충격성 하중 또는 저온 조건에서의 취성 파단 가능성")
            must.append(("cause", "저온·충격 조건이므로 연성-취성 천이(DBTT) 아래에서 파단되었을 가능성을 쓴다."))
        if hydrogen:
            causes.append("재료 취화(열처리·환경)에 따른 저에너지 파단 가능성")
            must.append(("cause", "수소 노출 조건이므로 수소에 의해 딤플 파괴가 준벽개·벽개로 바뀌었을 가능성(수소 취화)을 쓴다."))
            steps.append({"title": "수소 함량·표면 취성층 확인",
                          "detail": "확산성 수소 함량을 측정하고, 표면에서 내부로 갈수록 파면이 연성으로 바뀌는지 촬영 위치를 달리해 확인하세요."})
        if load == "cyclic":
            must.append(("cause", "반복 하중 조건이므로 피로 균열이 자란 뒤 최종 파단이 취성으로 일어났을 가능성도 쓴다."))
        if temp == "room" and load in ("static", "sustained") and not hydrogen and not corrosive:
            notes.append("상온·정하중·일반 환경에서 벽개로 분류되었습니다. 재질 취화나 오분류 가능성을 함께 확인하세요.")

    # ── 피로 ──
    elif pred_en == "Fatigue":
        if load in ("static", "impact", "sustained"):
            must.append(("basis", f"입력된 하중 형태({background_info.LOADS[load].split(' (')[0]})는 반복 하중이 아니므로 피로 판정과 맞지 않는다. "
                                  "슬립 흔적·쌍정·조직 무늬처럼 피로 줄무늬로 보이는 평행 무늬일 가능성을 밝힌다."))
            notes.append("반복 하중이 없는 조건에서 피로로 분류되었습니다. 줄무늬처럼 보이는 다른 무늬(슬립선·조직)일 수 있어 오분류 가능성이 있습니다.")
            steps.append({"title": "평행 무늬의 정체 확인",
                          "detail": "무늬 간격이 균열 진행 방향으로 규칙적으로 변하는지, 결정립마다 방향이 바뀌는지(슬립선) 고배율로 확인하세요."})
        if corrosive and load == "cyclic":
            causes.append("진동 등 운전 중 반복 응력의 영향 가능성")
            must.append(("cause", "부식 환경과 반복 하중이 함께 있으므로 부식 피로 가능성(부식 피트에서 시작, 줄무늬가 부식으로 흐려짐)을 쓴다."))
        if temp == "high":
            must.append(("cause", "고온 조건이므로 산화·크리프가 피로 균열 성장에 관여했을 가능성을 쓴다."))
        if "welded" in proc:
            causes.append("응력 집중부(노치·결함)에서 시작된 피로 균열 가능성")
            must.append(("cause", "용접 이력이 있으므로 용접 결함·잔류 응력이 피로 균열 시작에 관여했을 가능성을 쓴다."))
        if manner == "sudden":
            must.append(("cause", "전조 없이 파손된 것으로 입력되었지만, 피로는 균열이 오래 자란 뒤 마지막 파단만 순간적으로 일어날 수 있다고 쓴다."))

    # ── 입계 ──
    elif pred_en == "Intergranular":
        if hydrogen:
            causes.append("수소 또는 부식 환경에 의한 환경 보조 입계 균열 가능성")
            src = "도금·산세 공정" if "plated" in proc and "hydrogen" not in env else "수소 노출"
            must.append(("cause", f"{src} 이력이 있으므로 수소 취화에 의한 입계 균열 가능성을 가장 먼저 쓴다."))
            steps.append({"title": "수소 취화 확인",
                          "detail": "확산성 수소 함량 분석, 도금 후 베이킹 여부 확인, 입계면의 미세 딤플·머리카락 모양 선 유무를 고배율로 확인하세요."})
        if corrosive:
            causes.append("수소 또는 부식 환경에 의한 환경 보조 입계 균열 가능성")
            must.append(("cause", "부식 환경이므로 응력부식균열(SCC) 가능성을 쓴다. 파면의 부식 생성물과 가지 모양 2차 균열이 단서라고 덧붙인다."))
            steps.append({"title": "부식 생성물 성분 분석",
                          "detail": "파면과 2차 균열 내부를 EDS로 분석해 염소·황 등 부식 원소가 검출되는지 확인하세요."})
        if temp == "high":
            causes.append("고온 노출에 따른 입계 손상 가능성")
            must.append(("cause", "고온 조건이므로 크리프 입계 손상(입계 공동·쐐기 균열) 가능성을 쓴다."))
        if "heat_treated" in proc and not hydrogen and not corrosive:
            causes.append("결정립계 취화(편석·석출)에 따른 입계 분리 가능성")
            must.append(("cause", "열처리 이력이 있으므로 템퍼 취화·예민화처럼 입계 편석·석출에 의한 취화 가능성을 쓴다."))
        if not (hydrogen or corrosive or temp == "high" or proc):
            must.append(("cause", "입력된 조건만으로는 입계 취화 원인을 좁히기 어렵다는 점을 밝히고, 재료 내부 편석·석출 가능성을 쓴다."))

    # ── 공통: 파손 양상 ──
    if manner == "gradual" and pred_en in ("Ductile", "Cleavage"):
        must.append(("cause", "파손 전에 전조가 있었다면 피로·응력부식 같은 점진적 균열이 먼저 진행됐을 가능성이 있다고 쓴다."))

    # 중복 제거 (순서 유지)
    seen = set()
    causes = [c for c in causes if not (c in seen or seen.add(c))]
    return must, causes, notes, steps


def format_must_lines(must_lines: List[Tuple[str, str]]) -> str:
    return "\n".join(f"- ({field}) {text}" for field, text in must_lines)