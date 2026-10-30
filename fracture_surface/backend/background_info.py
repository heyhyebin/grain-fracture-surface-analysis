"""
사전 배경 정보 (사용자 입력, 모두 선택 사항)

실무 고장 분석 1단계(배경 정보 수집) 중 원인 판단에 기여도가 큰 항목만 받는다.

- environment   : 사용 환경 (복수)  air | chloride | hydrogen | chemical      — 쉼표로 구분
- load          : 하중 형태          static | impact | cyclic | sustained
- temperature   : 사용 온도          low | room | high
- process       : 제조·처리 이력 (복수) heat_treated | plated | welded | cold_worked — 쉼표로 구분
- failure_manner: 파손 양상          sudden | gradual
- note          : 자유 메모 (사용 기간, 시험 조건 등, 최대 200자)

빈 값·알 수 없는 값은 모두 "모름"으로 처리한다. 모든 항목이 모름이면
보고서와 카드 선택은 배경 정보가 없던 때와 똑같이 동작한다.
"""

from typing import Dict, List, Optional

ENVIRONMENTS = {
    "air": "일반 대기",
    "chloride": "수분·염화물 (해수 등)",
    "hydrogen": "수소 노출 (수소 가스·수소 충전·전기도금·산세·음극방식·H₂S)",
    "chemical": "산·알칼리·황화물 등 화학 약품",
}

LOADS = {
    "static": "정하중·단조 인장 (과부하, 인장 시험 포함)",
    "impact": "충격 하중",
    "cyclic": "반복·진동 하중",
    "sustained": "장시간 지속 하중",
}

TEMPERATURES = {
    "low": "저온 (0 ℃ 이하·극저온)",
    "room": "상온",
    "high": "고온 (수백 ℃ 이상)",
}

PROCESSES = {
    "heat_treated": "열처리",
    "plated": "도금·산세",
    "welded": "용접",
    "cold_worked": "냉간 가공",
}

MANNERS = {
    "sudden": "전조 없이 갑자기 파손",
    "gradual": "전조 있음 (균열 성장·소음·누설 등)",
}

NOTE_MAX = 200

KEYS = ("environment", "load", "temperature", "process", "failure_manner", "note")


def _pick(value, allowed) -> str:
    value = str(value or "").strip()
    return value if value in allowed else ""


def _pick_many(value, allowed) -> List[str]:
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = str(value or "").replace(";", ",").split(",")
    out = []
    for item in items:
        item = str(item).strip()
        if item in allowed and item not in out:
            out.append(item)
    return out


def normalize(raw: Optional[Dict] = None) -> Dict:
    """프론트·저장 기록에서 온 값을 정리한다. 모르는 값은 "" / [] 로 둔다."""
    raw = raw or {}
    note = " ".join(str(raw.get("note") or "").split())[:NOTE_MAX]
    return {
        "environment": _pick_many(raw.get("environment"), ENVIRONMENTS),
        "load": _pick(raw.get("load"), LOADS),
        "temperature": _pick(raw.get("temperature"), TEMPERATURES),
        "process": _pick_many(raw.get("process"), PROCESSES),
        "failure_manner": _pick(raw.get("failure_manner"), MANNERS),
        "note": note,
    }


def is_empty(info: Optional[Dict]) -> bool:
    info = info or {}
    return not any(info.get(k) for k in KEYS)


def items(info: Optional[Dict]):
    """(항목명, 값) 목록. 입력된 항목만."""
    info = info or {}
    out = []
    if info.get("environment"):
        out.append(("사용 환경", ", ".join(ENVIRONMENTS[e] for e in info["environment"])))
    if info.get("load"):
        out.append(("하중 형태", LOADS[info["load"]]))
    if info.get("temperature"):
        out.append(("사용 온도", TEMPERATURES[info["temperature"]]))
    if info.get("process"):
        out.append(("제조·처리 이력", ", ".join(PROCESSES[p] for p in info["process"])))
    if info.get("failure_manner"):
        out.append(("파손 양상", MANNERS[info["failure_manner"]]))
    if info.get("note"):
        out.append(("사용자 메모", info["note"]))
    return out


def label(info: Optional[Dict]) -> str:
    """보고서 머리말용 한 줄 요약 (메모는 길어서 뺀다)."""
    pairs = [(k, v) for k, v in items(info) if k != "사용자 메모"]
    out = []
    for k, v in pairs:
        if k == "사용 환경":
            v = ", ".join(ENVIRONMENTS[e].split(" (")[0] for e in (info or {}).get("environment", []))
        elif "(" in v:
            v = v.split(" (")[0]
        out.append(f"{k} {v}")
    return " · ".join(out) if out else "모름"


def prompt_text(info: Optional[Dict]) -> str:
    """LLM 프롬프트용 목록. 입력이 없으면 빈 문자열."""
    pairs = items(info)
    if not pairs:
        return ""
    lines = [f"- {k}: {v}" for k, v in pairs]
    if (info or {}).get("note"):
        lines.append("- 사용자 메모는 사용자가 적은 내용이며, 확인된 사실로 단정하지 않는다.")
    return "\n".join(lines)