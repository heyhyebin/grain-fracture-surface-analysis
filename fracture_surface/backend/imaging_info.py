"""
SEM 촬영 정보 (사용자 입력, 모두 선택 사항)

- magnification : 배율 (예: 2000). 장비마다 기준 화면 크기가 달라 근사값으로만 쓴다.
- scale_bar_um  : 스케일 바 표시 값 (µm)
- location      : 촬영 위치  origin | propagation | final | overview
- detector      : 검출기     SE | BSE
- cleaned       : 세척 여부  yes | no
- coated        : 도전성 코팅 여부  yes | no

빈 값·알 수 없는 값은 모두 "모름"으로 처리한다. 모든 항목이 모름이면
보고서와 카드 선택은 촬영 정보가 없던 때와 똑같이 동작한다.
"""

from typing import Dict, Optional

LOCATIONS = {
    "origin": "균열 시작점 부근",
    "propagation": "균열 전파 영역",
    "final": "최종 파단부",
    "overview": "파면 전체 (저배율)",
}

DETECTORS = {
    "SE": "이차전자(SE)",
    "BSE": "후방산란전자(BSE)",
}

YES_NO = {"yes": "함", "no": "안 함"}

KEYS = ("magnification", "scale_bar_um", "location", "detector", "cleaned", "coated")


def _positive_number(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).replace(",", "").strip().rstrip("xX×배").strip()
    try:
        number = float(text)
    except ValueError:
        return None
    return number if number > 0 else None


def normalize(raw: Optional[Dict] = None) -> Dict:
    """프론트·저장 기록에서 온 값을 정리한다. 모르는 값은 None / "" 로 둔다."""
    raw = raw or {}

    def pick(key, allowed):
        value = str(raw.get(key) or "").strip()
        return value if value in allowed else ""

    return {
        "magnification": _positive_number(raw.get("magnification")),
        "scale_bar_um": _positive_number(raw.get("scale_bar_um")),
        "location": pick("location", LOCATIONS),
        "detector": pick("detector", DETECTORS),
        "cleaned": pick("cleaned", YES_NO),
        "coated": pick("coated", YES_NO),
    }


def has_scale(info: Optional[Dict]) -> bool:
    info = info or {}
    return bool(info.get("magnification") or info.get("scale_bar_um"))


def is_empty(info: Optional[Dict]) -> bool:
    info = info or {}
    return not any(info.get(k) for k in KEYS)


def _fmt(number: float) -> str:
    return f"{number:g}"


def items(info: Optional[Dict]):
    """(항목명, 값) 목록. 입력된 항목만."""
    info = info or {}
    out = []
    if info.get("magnification"):
        out.append(("배율", f"약 {_fmt(info['magnification'])}배"))
    if info.get("scale_bar_um"):
        out.append(("스케일 바", f"{_fmt(info['scale_bar_um'])} µm"))
    if info.get("location"):
        out.append(("촬영 위치", LOCATIONS[info["location"]]))
    if info.get("detector"):
        out.append(("검출기", DETECTORS[info["detector"]]))
    if info.get("cleaned"):
        out.append(("세척", YES_NO[info["cleaned"]]))
    if info.get("coated"):
        out.append(("도전성 코팅", YES_NO[info["coated"]]))
    return out


def label(info: Optional[Dict]) -> str:
    """보고서 머리말용 한 줄 요약."""
    pairs = items(info)
    return " · ".join(f"{k} {v}" for k, v in pairs) if pairs else "모름"


def prompt_text(info: Optional[Dict]) -> str:
    """LLM 프롬프트용 목록. 입력이 없으면 빈 문자열."""
    pairs = items(info)
    if not pairs:
        return ""
    lines = [f"- {k}: {v}" for k, v in pairs]
    if (info or {}).get("magnification") and not (info or {}).get("scale_bar_um"):
        lines.append("- 배율 숫자는 장비마다 기준이 달라 근사값이다.")
    return "\n".join(lines)