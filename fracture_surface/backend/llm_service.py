"""
[비교용] fallback 제거 버전
- 미리 정해 둔 대체 문장(fallback)을 모두 없앴다. LLM 이 낸 문장을 정리(clean_text)만 해서 그대로 쓴다.
- 검증 규칙은 문장을 바꾸지 않고 "[검증 경고]" 로그만 남긴다.
- LLM 호출이 실패하면 해당 필드는 빈 문자열이고, 응답에 llm_error 가 담긴다.

LLM 서비스 모듈 (Gemma / Ollama)

[프롬프트 개편]
- 분석 호출은 '관찰 전용'에 가깝게 줄였다. RAG 문헌은 기본으로 넣지 않는다 (USE_RAG_IN_ANALYSIS).
- 촬영 위치·스케일·재질·2위 확률은 report_rules.py 가 먼저 판단해 [필수 반영 사항]과 원인 후보로 넣는다.
- 스케일 바가 있으면 수치 단정은 막고 '스케일 바 대비 상대 크기' 서술은 요구한다.
- 모든 호출은 _chat() 으로: think=False, JSON 스키마, 토큰 수 로그.
- 보고서 used_cards 는 본문에 카드 내용이 실제로 드러난 경우만 '인용'으로 남긴다.

- generate_llm_analysis()     : 단일 이미지 멀티모달 분석 (/predict)
- generate_compare_analysis() : 두 분석 결과 비교
- generate_report()           : 파손단면 AI 분석 보고서 (/report)
"""

import os
import re
import json
import ollama
from datetime import datetime

import background_info
import imaging_info
import report_rules
from knowledge_cards import (
    format_cards_for_prompt,
    parse_used_cards,
    public_cards,
    select_cards,
    verify_used_cards,
)

# =========================================================
# Ollama 모델 설정
# =========================================================

# 환경변수로 바꿀 수 있다. 예) PowerShell: $env:OLLAMA_MODEL = "gemma4:e4b"
# 원격(Colab 등) Ollama 를 쓰려면 $env:OLLAMA_HOST = "https://....trycloudflare.com"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:12b")

# 분석·비교·보고서가 같은 값을 써야 Ollama 가 호출마다 모델을 다시 올리지 않는다.
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))

# 분석(관찰) 호출에 RAG 문헌 청크를 넣을지. 기본 끔:
# 클래스 이름만으로 검색해 매번 같은 영어 청크가 들어가고, 이미지 관찰 지시를 묻어 버린다.
# 문헌 근거는 보고서 단계의 지식 카드(같은 문헌의 한국어 요약)로 대신한다.
USE_RAG_IN_ANALYSIS = os.getenv("USE_RAG_IN_ANALYSIS", "0") == "1"


def _chat(**kwargs):
    """
    ollama.chat 공통 래퍼.
    - think=False: gemma4·qwen3.5 의 사고 모드가 출력 토큰을 다 써서 JSON 이 비는 것을 막는다.
      (구버전 ollama 패키지가 think 인자를 모르면 빼고 다시 호출)
    - 입력/출력 토큰 수를 로그로 남겨 num_ctx 초과 여부를 확인할 수 있게 한다.
    """
    kwargs.setdefault("model", OLLAMA_MODEL)
    try:
        response = ollama.chat(think=False, **kwargs)
    except TypeError:
        response = ollama.chat(**kwargs)

    prompt_tokens = response.get("prompt_eval_count")
    print(
        f"[LLM] {kwargs['model']} 입력 {prompt_tokens} / num_ctx {LLM_NUM_CTX} 토큰, "
        f"출력 {response.get('eval_count')} 토큰, 종료 사유 {response.get('done_reason')}"
    )
    if prompt_tokens and prompt_tokens >= LLM_NUM_CTX * 0.95:
        print("[LLM 경고] 입력이 num_ctx 에 거의 닿았습니다. 앞부분이 잘렸을 수 있으니 LLM_NUM_CTX 를 늘리세요.")
    return response


# 구조화 출력(JSON 스키마) — 필드 누락·엉뚱한 키를 줄인다.
ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "feature": {"type": "string"},
        "visual_observation_1": {"type": "string"},
        "visual_observation_2": {"type": "string"},
        "mechanism_relation": {"type": "string"},
        "expected_cause": {"type": "string"},
        "cause_explanation": {"type": "string"},
    },
    "required": ["feature", "visual_observation_1", "visual_observation_2",
                 "mechanism_relation", "expected_cause", "cause_explanation"],
}

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "characteristics": {"type": "string"},
        "basis": {"type": "string"},
        "cause": {"type": "string"},
        "next_steps_note": {"type": "string"},
        "used_cards": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["characteristics", "basis", "cause", "next_steps_note", "used_cards"],
}

# =========================================================
# 공통 상수
# =========================================================

FRACTURE_TYPES = ["연성 파괴", "취성 파괴", "피로 파괴", "입계 파괴"]

KO_NAMES = {
    "Cleavage": "취성 파괴",
    "Ductile": "연성 파괴",
    "Fatigue": "피로 파괴",
    "Intergranular": "입계 파괴",
}

MATERIAL_LABELS = {
    "steel": "강",
    "stainless_steel": "스테인리스강",
    "aluminum": "알루미늄",
    "titanium": "티타늄",
    "cast_iron": "주철",
    "copper": "구리",
    "magnesium": "마그네슘 합금",
    "nickel_alloy": "니켈 합금",
    "tool_steel": "공구강",
    "hea": "고엔트로피 합금 (HEA)",
    "unknown": "정보 없음",
    "": "미선택",
}

UNKNOWN_MATERIALS = ("", "unknown", None)

# =========================================================
# 분석 프롬프트용 원인 후보
# =========================================================

# 유형별 기본 원인 후보. 실제 프롬프트에는 report_rules.build_context() 가
# 촬영 위치·재질에 맞춰 순서를 바꾼 후보가 들어간다.
CAUSE_CANDIDATES = {
    KO_NAMES[en]: causes for en, causes in report_rules.BASE_CAUSES.items()
}

# =========================================================
# 보고서용 상수
# =========================================================

MATERIAL_TRAITS = {
    "steel": "강도와 인성의 균형이 좋지만 탄소량과 열처리에 따라 연성-취성 천이 온도가 나타날 수 있고, 반복 하중에서는 피로 한도가 존재하는 재료입니다.",
    "stainless_steel": "오스테나이트계는 인성이 높고 저온에서도 연성을 유지하지만 염화물 환경에서 응력부식균열에 민감할 수 있으며, 마르텐사이트계는 경도가 높은 대신 상대적으로 취성이 큽니다.",
    "aluminum": "가볍고 연성이 좋지만 뚜렷한 피로 한도가 없어 반복 하중에서 장기적으로 피로 균열이 발생할 수 있습니다.",
    "titanium": "비강도가 높지만 노치(응력 집중부)에 민감하고 수소를 흡수하면 취화될 수 있습니다.",
    "cast_iron": "내부 흑연 조직 때문에 인성이 낮고 인장에 약해 취성 파괴가 일어나기 쉬운 재료입니다.",
    "copper": "연성과 전도성이 높아 연성 파괴 경향이 크며, 조건에 따라 수소 취화가 문제될 수 있습니다.",
    "magnesium": "상온 성형성이 낮고 노치에 민감하며 부식과 피로에 상대적으로 취약한 편입니다.",
    "nickel_alloy": "고온 강도와 내식성이 우수하지만 고온 환경에서는 입계 석출물, 크리프와 관련된 입계 손상이 문제될 수 있습니다.",
    "tool_steel": "경도와 내마모성이 높은 대신 인성이 낮아 충격이나 응력 집중에 취성 파괴가 일어나기 쉽습니다.",
    "hea": "FeMnNiCrCo(Cantor) 같은 단상 FCC 고엔트로피 합금은 저온에서도 연성과 인성을 유지하는 편이지만, 수소 충전 시 쌍정·입계를 따라 균열이 생기는 수소 취화가 보고됩니다.",
}

NEXT_STEPS = {
    "취성 파괴": [
        {"title": "사용 온도와 충격 이력 확인", "detail": "파손 당시의 온도, 충격이나 급격한 하중 여부를 확인하세요. 저온이나 충격 조건은 취성 파괴의 주요 요인입니다."},
        {"title": "균열 시작점 추적", "detail": "파단면의 방사형(헤링본) 무늬를 따라 균열 시작점을 찾고, 시작점 주변에 노치, 용접부, 가공 흔적이 있는지 확인하세요."},
        {"title": "충격 시험", "detail": "샤르피 충격 시험으로 인성과 연성-취성 천이 온도를 확인하면 재료 자체의 취약 여부를 판단할 수 있습니다."},
        {"title": "미세조직 및 경도 확인", "detail": "단면 연마 후 조직 관찰과 경도 측정으로 열처리 상태, 개재물, 조직 불균일 여부를 확인하세요."},
    ],
    "연성 파괴": [
        {"title": "하중 이력 확인", "detail": "설계 하중을 초과한 과부하나 사고 기록이 있었는지 확인하세요."},
        {"title": "파단부 거시 관찰", "detail": "네킹(단면 수축), 컵-콘 형상, 전단립 방향을 확인하면 하중 방향과 변형 정도를 추정할 수 있습니다."},
        {"title": "SEM 딤플 관찰", "detail": "딤플의 크기와 형태(등축, 전단, 인장)를 관찰하면 하중 형태를 구분하는 데 도움이 됩니다."},
        {"title": "인장 시험과 단면 확인", "detail": "해당 재료의 강도와 연신율이 규격에 부합하는지, 실제 단면이 설계보다 줄어 있지 않은지 확인하세요."},
    ],
    "피로 파괴": [
        {"title": "균열 기점과 비치 마크 확인", "detail": "비치 마크가 시작되는 기점을 찾고, 기점이 표면인지 내부인지, 주변에 노치나 결함이 있는지 확인하세요."},
        {"title": "하중 이력 조사", "detail": "반복 횟수, 진동, 응력 진폭 등 운전 조건을 확인하세요. 최종 파단 영역의 비율로 하중 수준을 가늠할 수 있습니다."},
        {"title": "표면 상태 점검", "detail": "가공 흔적, 부식 피트, 용접부, 잔류 응력 등 기점 후보가 될 수 있는 표면 조건을 확인하세요."},
        {"title": "SEM 스트리에이션 관찰", "detail": "미세 줄무늬(스트리에이션)를 관찰하면 피로 파괴 여부를 더 확실하게 검증할 수 있습니다."},
    ],
    "입계 파괴": [
        {"title": "사용 환경 확인", "detail": "수소, 염화물이나 황화물 같은 부식 매질, 고온 노출 이력이 있었는지 확인하세요."},
        {"title": "SEM-EDS 성분 분석", "detail": "입계면에서 석출물이나 편석 원소(예: S, P 등)가 검출되는지 확인하세요."},
        {"title": "금속 조직 관찰", "detail": "단면을 연마하고 에칭해 결정립 크기, 입계 석출물, 조직 이상을 확인하세요."},
        {"title": "열처리·용접 이력 확인", "detail": "예민화, 템퍼 취화 등 입계를 약화시킬 수 있는 열처리나 용접 이력을 확인하세요."},
    ],
}

FORBIDDEN_DEFINITIVE = ["확실합니다", "분명합니다", "원인입니다", "틀림없습니다"]

MATERIAL_WORDS = [
    "재질", "스테인리스", "알루미늄", "티타늄", "주철",
    "구리", "마그네슘", "니켈", "공구강", "강재", "탄소강", "고엔트로피",
]


# =========================================================
# 텍스트 처리
# =========================================================

def strip_empty_markers(text: str) -> str:
    """Remove accidental literal placeholders without altering legitimate observations."""
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"[\(（\[【]\s*빈\s*문자열\s*[\)）\]】]", "", text)
    text = re.sub(r"(?<![가-힣A-Za-z])빈\s*문자열(?![가-힣A-Za-z])", "", text)
    text = re.sub(r"\s+([.,!?。])", r"\1", text)
    text = re.sub(r"([.!?。])(?:\s*[.!?。])+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    if text is None:
        return ""

    text = strip_empty_markers(text)

    substitutions = (
        (r"\bjagged\b", "불규칙하고 각진"),
        (r"\birregular\b", "불규칙한"),
        (r"\brough\b", "거친"),
        (r"\bhexagon(?:al)?\b", "다각형"),
    )
    for pattern, replacement in substitutions:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = text.replace("다각형 모양", "다각형")
    text = re.sub(r"\*+|#", "", text)
    text = re.sub(r"^\s*\d+\.\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip()


def warn(field: str, reason: str) -> None:
    """검증 규칙에 걸려도 문장을 바꾸지 않고 로그만 남긴다."""
    print(f"[검증 경고] {field}: {reason}")


def ensure_sentence_end(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    if text[-1] not in ".!?。":
        text += "."
    return text


def parse_llm_json(text: str):
    if not text:
        return None

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            obj = json.loads(match.group())
            return obj if isinstance(obj, dict) else None
        except (ValueError, TypeError):
            pass

    return None


# =========================================================
# 신뢰도
# =========================================================

def get_confidence_level(confidence_percent: float) -> str:
    if confidence_percent < 50:
        return "low"
    elif confidence_percent < 75:
        return "medium"
    return "high"


def get_confidence_instruction(confidence_percent: float) -> str:
    level = get_confidence_level(confidence_percent)

    if level == "low":
        return (
            "CNN 신뢰도가 낮다. "
            "파손 유형과 이미지 특징의 관계를 강하게 단정하지 않는다. "
            "'관련될 가능성이 있습니다', "
            "'해석할 수 있습니다'와 같은 표현을 사용한다. "
            "최종 설명에 추가 검토 필요성을 포함한다."
        )

    if level == "medium":
        return (
            "CNN 신뢰도가 중간 수준이다. "
            "확정적인 표현을 피하고 "
            "'관련될 가능성이 있습니다', "
            "'해석할 수 있습니다'와 같이 작성한다."
        )

    return (
        "CNN 신뢰도가 비교적 높더라도 "
        "이미지만으로 파손 메커니즘이나 원인을 확정하지 않는다. "
        "'관련성이 높아 보입니다', "
        "'해석할 수 있습니다', "
        "'가능성이 있습니다'와 같이 작성한다."
    )


# =========================================================
# 재질 입력 / Grad-CAM 텍스트
# =========================================================

def build_condition_text(material: str) -> str:
    if material in UNKNOWN_MATERIALS:
        return "재질 정보 없음"

    material_ko = MATERIAL_LABELS.get(material, material)
    return (
        f"사용자가 입력한 참고 재질: {material_ko}\n"
        "재질은 참고 메타데이터일 뿐이며 "
        "이미지 관찰, 주요 특징, 파손 메커니즘, "
        "파손 원인을 결정하는 근거로 사용하지 않는다."
    )


def build_cam_text(gradcam_summary, probabilities, decision_path):
    lines = []

    if decision_path:
        lines.append(f"최종 판정 방식: {decision_path}")

    if probabilities:
        lines.append("클래스별 예측 확률 (Grad-CAM에서 검출되지 않은 유형은 '-'):")
        for name, value in probabilities.items():
            lines.append(f"- {name}: {value}")

    if not gradcam_summary:
        lines.append("Grad-CAM++ 활성화 영역: 검출된 영역 없음")
        return "\n".join(lines)

    lines.append("")
    lines.append("Grad-CAM++ 활성화 영역 (모델이 판단에 활용한 영역):")

    for en_name, info in gradcam_summary.items():
        overlaps = ", ".join(
            KO_NAMES.get(o, o) for o in info["overlaps_with"]
        ) or "없음"
        lines.append(
            f"- {KO_NAMES.get(en_name, en_name)}: "
            f"확률 {info['probability']}%, "
            f"주 위치 {info['position']}, "
            f"{info['spread']}, "
            f"영역 {info['region_count']}개, "
            f"다른 유형과 겹침 {info['overlap_ratio']}% ({overlaps})"
        )

    return "\n".join(lines)


# =========================================================
# 단일 분석 Prompt
# =========================================================

MECHANISM_HINT = {
    "연성 파괴": "소성 변형과 미세 공극의 생성·성장·합체",
    "취성 파괴": "소성 변형이 적은 상태에서 결정면을 따라 분리",
    "피로 파괴": "반복 하중에 따른 점진적 균열 성장 (관찰만으로 반복 하중을 확인했다고 쓰지 않는다)",
    "입계 파괴": "결정립 경계를 따른 분리",
}


def build_prompt(prediction: str, confidence_percent: float, material: str,
                 imaging=None, cam_text: str = "", context=None,
                 references_text: str = "", background=None) -> str:
    """
    분석(관찰) 프롬프트.
    - 이 호출의 핵심은 '이미지에서 보이는 것'을 기록하는 것이다. 해석·원인은 짧게 두고,
      보고서 단계에서 지식 카드와 필수 반영 사항으로 확장한다.
    - 금지 규칙을 줄이고 '무엇을 쓰는지'를 구체적으로 지시했다 (금지가 많으면 작은 모델은 일반론으로 도망간다).
    - 촬영 정보·관찰 지시는 JSON 형식보다 앞에 둔다.
    """
    context = context or {}
    imaging_text = imaging_info.prompt_text(imaging) or "- 입력 없음"
    background_text = background_info.prompt_text(background) or "- 입력 없음"
    material_text = (
        "정보 없음" if material in UNKNOWN_MATERIALS
        else f"{MATERIAL_LABELS.get(material, material)} (관찰에는 쓰지 않고, 원인 후보를 고를 때만 참고)"
    )
    obs_hints = context.get("obs_hints") or [
        "스케일 정보가 없으므로 크기는 '이미지 폭에 비해 작다/크다'처럼 이미지 안에서의 상대 크기로만 쓴다."
    ]
    hints = "\n".join(f"- {h}" for h in obs_hints)
    causes = "\n".join(f"- {c}" for c in (context.get("cause_candidates") or CAUSE_CANDIDATES[prediction]))
    refs = ""
    if references_text:
        refs = ("\n[참고 문헌 발췌] (일반 지식이며 이 이미지의 관찰 사실이 아니다)\n"
                + str(references_text)[:2500] + "\n")

    return f"""너는 금속 파단면 SEM 이미지를 관찰해 기록하는 도우미다.
첫 번째 이미지는 원본이다. 두 번째 이미지가 있으면 같은 이미지에 CNN 이 주목한 영역의 윤곽선을 그린 것이다.

[입력 정보]
- CNN 예측: {prediction} (확률 {confidence_percent:.1f}%). 이 예측은 바꾸지 않는다.
- 재질: {material_text}
- SEM 촬영 정보:
{imaging_text}
- 사전 배경 정보 (관찰에는 쓰지 않고, 원인 후보를 고를 때만 참고):
{background_text}

[CNN 판단 정보]
{cam_text or "- 없음"}
{refs}
[관찰 지시]
- 원본 이미지에서 실제로 보이는 형상(함몰부, 평탄한 면, 계단, 선형 무늬, 경계, 2차 균열 등)의 모양·상대 크기·밀도·분포를 쓴다.
{hints}
- 한 이미지 안에 형태가 다른 영역이 섞여 있으면 각각 어디에 있는지 쓴다.
- 윤곽선의 색, 이미지 속 글자·스케일 바·화살표·패널 기호는 파면 특징이 아니다.
- 확실하지 않으면 "~로 보인다"로 쓰고, 보이지 않는 것은 쓰지 않는다.

[작성할 필드]
- feature: 가장 눈에 띄는 형상을 20~50자 명사구로 쓴다. 원인·파손 유형 이름·재질은 넣지 않는다.
  (예: "스케일 바보다 훨씬 작은 등축 딤플이 밀집한 표면")
- visual_observation_1: 대표 형상의 모양·상대 크기·밀도를 1~2문장으로 쓴다.
- visual_observation_2: 그 형상의 분포 위치, 다른 형태의 영역이 있는지를 1~2문장으로 쓴다.
- mechanism_relation: 관찰한 형상을 {prediction} 메커니즘({MECHANISM_HINT[prediction]})과 1문장으로 연결한다. "관련지어 해석할 수 있습니다"처럼 쓴다.
- expected_cause: 아래 후보 중 하나를 글자 그대로 고른다. 판단할 근거가 없으면 첫 번째 후보를 고른다.
{causes}
- cause_explanation: 고른 원인을 관찰·재질·사전 배경 정보와 연결해 추정형("가능성이 있습니다")으로 1~2문장 쓴다.

{get_confidence_instruction(confidence_percent)}
반드시 한국어로, 아래 JSON 객체만 출력한다.
{{
  "feature": "...",
  "visual_observation_1": "...",
  "visual_observation_2": "...",
  "mechanism_relation": "...",
  "expected_cause": "...",
  "cause_explanation": "..."
}}"""


# =========================================================
# 단일 분석 응답 검증 (경고만, 문장 대체 없음)
# =========================================================

def validate_feature(prediction: str, feature: str) -> str:
    """해석절만 잘라내고, 규칙에 걸려도 대체하지 않는다."""
    feature = clean_text(feature).strip().rstrip('.!?。')

    if not feature:
        warn("feature", "빈 응답")
        return ""

    feature = re.sub(r'^(?:이미지에서|이미지에는|파단면에서|파단면에는)\s*', '', feature)

    # 해석절은 잘라내고 관찰절만 남긴다.
    feature = re.split(r'[,，]?\s*(?:이는|이러한 구조는|이러한 특징은)\s*(?=.{0,100}(?:파괴|메커니즘|응력|하중|관련))', feature, maxsplit=1)[0]

    if re.search(r'CNN|예측|응력|하중|원인|메커니즘|파괴|\d+\s*(?:mm|μm|cm)', feature):
        warn("feature", "원인·유형·수치 표현 포함")

    if re.search(r'붉은색|푸른색|빨간색|파란색|색상|밝기', feature):
        warn("feature", "색상·밝기 표현 포함")

    if len(feature) < 4:
        warn("feature", "너무 짧음")

    return feature


def validate_visual_observation(text: str) -> str:
    """원인 해석만 잘라내고 가능한 관찰 절을 보존한다."""
    text = clean_text(text)
    if not text:
        return ''

    results = []
    for sentence in re.split(r'(?<=[.!?。])\s+', text):
        # 이미지에서 잰 것처럼 보이는 수치는 빼되, 스케일 바와의 상대 비교 문장은 남긴다.
        if (re.search(r'\d+(?:\.\d+)?\s*(?:mm|cm|μm|µm|㎛|nm)', sentence, re.I)
                and '스케일' not in sentence):
            continue

        # 인과·분류 해석 접속절만 제거한다.
        sentence = re.split(
            r'[,，]?\s*(?:이는|이러한\s*(?:구조|분포|특징|형태)는|그 결과)\s*'
            r'(?=.{0,130}(?:파괴|CNN|예측|메커니즘|응력|하중|원인|시사|기여|유발|전파))',
            sentence, maxsplit=1
        )[0]
        sentence = re.split(r'[,，]?\s*(?:시간이 지나면서|점진적으로|점차)\s*(?:성장|확장|전파|진행)', sentence, maxsplit=1)[0]

        if re.search(r'CNN|예측|메커니즘|응력 집중|하중 집중|경계 약화|반복 하중|반복적인 하중|파괴 확산', sentence):
            continue

        sentence = sentence.strip().rstrip('.!?。 ')
        for a, b in [('이어져 있으며', '이어져 있습니다'), ('나타나며', '나타납니다'),
                     ('관찰되며', '관찰됩니다'), ('보이며', '보입니다'), ('있으며', '있습니다')]:
            if sentence.endswith(a):
                sentence = sentence[:-len(a)] + b
                break

        if sentence and not sentence.endswith(('하며', '이고', '이고서', '때문에', '통해')):
            results.append(ensure_sentence_end(sentence))

    return ' '.join(results)


def validate_mechanism_relation(prediction: str, text: str) -> str:
    text = clean_text(text)

    if not text:
        warn("mechanism_relation", "빈 응답")
        return ""

    if any(name != prediction and name in text for name in FRACTURE_TYPES):
        warn("mechanism_relation", "다른 파손 유형명 포함")

    if re.search(r'확실|입증|증명|측정|반드시|응력 집중|하중 집중|반복(?:적인)? 하중|경계 약화|취약화|유발|가속화|촉진', text):
        warn("mechanism_relation", "단정 또는 원인 추정 표현 포함")

    terms = {
        '연성 파괴': ('소성 변형', '공극', '변형을 동반'),
        '취성 파괴': ('균열 전파', '균열 진행', '소성 변형', '분리'),
        '피로 파괴': ('균열 성장', '균열 전파', '점진적'),
        '입계 파괴': ('경계를 따라', '경계를 따른', '경계 분리', '입계 분리', '분리 과정'),
    }
    if not any(term in text for term in terms[prediction]):
        warn("mechanism_relation", "예측 유형의 메커니즘 용어 없음")

    return ensure_sentence_end(text)


def validate_expected_cause(prediction: str, cause: str, candidates=None) -> str:
    cause = clean_text(cause)

    if not cause:
        warn("expected_cause", "빈 응답")
        return ""

    if cause.rstrip('.!?。 ') not in (candidates or CAUSE_CANDIDATES[prediction]):
        warn("expected_cause", "허용 후보 목록에 없는 문구")

    invalid_keywords = {
        "연성 파괴": ["반복 하중", "반복적인 하중", "반복 응력", "반복적인 응력", "피로", "결정립 경계"],
        "취성 파괴": ["반복 하중", "반복적인 하중", "반복 응력", "반복적인 응력", "피로", "큰 소성 변형"],
        "피로 파괴": ["단일 하중", "한 번의 큰 하중", "결정립 경계 약화", "과도한 소성 변형"],
        "입계 파괴": ["반복 하중", "반복적인 하중", "반복 응력", "반복적인 응력", "피로", "큰 소성 변형"],
    }
    for word in invalid_keywords.get(prediction, []):
        if word in cause:
            warn("expected_cause", f"부적절 원인 키워드: {word}")

    return cause


def validate_cause_explanation(prediction: str, text: str) -> str:
    text = clean_text(text)

    if not text:
        warn("cause_explanation", "빈 응답")
        return ""

    if any(name != prediction and name in text for name in FRACTURE_TYPES):
        warn("cause_explanation", "다른 파손 유형명 포함")

    invalid_keywords = {
        '연성 파괴': ('반복 하중', '반복적인 하중', '반복 응력', '결정립 경계'),
        '취성 파괴': ('반복 하중', '반복적인 하중', '반복 응력', '큰 소성 변형'),
        '피로 파괴': ('한 번의 큰 하중', '급격한 단일 하중', '결정립 경계 약화'),
        '입계 파괴': ('반복 하중', '반복적인 하중', '반복 응력', '큰 소성 변형'),
    }
    if any(word in text for word in invalid_keywords[prediction]):
        warn("cause_explanation", "예측 유형과 맞지 않는 원인 키워드")

    surface = r'(?:구조|형태|무늬|균열|네트워크|틈새|분포|파편|공극|함몰|간격|층상)'
    cause = r'(?:응력|하중|충격|경계 약화|취약화)'

    if (re.search(surface + r'.{0,65}' + cause + r'.{0,25}(?:유발|발생시켰|만들었)', text)
            or re.search(surface + r'.{0,75}' + cause + r'.{0,30}(?:발생|형성|파괴|유발|생겨|만들)', text)
            or re.search(cause + r'.{0,70}' + surface + r'.{0,35}(?:발생|형성|파괴|유발|생겨|만들)', text)):
        warn("cause_explanation", "표면 구조와 원인 사이의 과도한 인과 귀속")

    if not re.search(r'가능성|추정|고려될 수|해석될 수|영향을 미쳤을 수|작용했을 수', text):
        warn("cause_explanation", "추정형 표현 없음")

    if re.search(r'확실|분명|입증|증명|측정|원인입니다', text):
        warn("cause_explanation", "단정 표현 포함")

    return ensure_sentence_end(text)


def compose_explanation(prediction: str, confidence_percent: float,
                        visual_observation_1: str, visual_observation_2: str,
                        mechanism_relation: str, cause_explanation: str) -> str:
    sentences = [
        ensure_sentence_end(visual_observation_1),
        ensure_sentence_end(visual_observation_2),
        ensure_sentence_end(mechanism_relation),
        ensure_sentence_end(cause_explanation),
    ]

    if get_confidence_level(confidence_percent) == "low":
        sentences.append(
            "CNN 신뢰도가 낮아 추가 이미지나 "
            "전문가 검토를 함께 고려하는 것이 좋습니다."
        )

    return " ".join(s for s in sentences if s).strip()


# =========================================================
# 단일 이미지 Gemma 멀티모달 분석
# =========================================================

def generate_llm_analysis(
    prediction: str,
    confidence_percent: float,
    material: str = "",
    image_bytes: bytes = None,
    gradcam_image_bytes: bytes = None,
    gradcam_summary=None,
    probabilities=None,
    decision_path: str = "",
    references_text: str = "",
    imaging=None,
    background=None,
):
    imaging = imaging_info.normalize(imaging)
    background = background_info.normalize(background)
    prediction_en = next((en for en, ko in KO_NAMES.items() if ko == prediction), prediction)
    context = report_rules.build_context(prediction_en, material, imaging, probabilities, background)

    prompt = build_prompt(
        prediction, confidence_percent, material,
        imaging=imaging,
        cam_text=build_cam_text(gradcam_summary, probabilities, decision_path),
        context=context,
        references_text=references_text if USE_RAG_IN_ANALYSIS else "",
        background=background,
    )

    parsed = None
    llm_error = ""

    try:
        user_message = {"role": "user", "content": prompt}

        if image_bytes is not None:
            images = [image_bytes]
            if gradcam_image_bytes is not None:
                images.append(gradcam_image_bytes)
                print("[Gemma] 원본 이미지 + Grad-CAM++ 이미지 입력")
            else:
                print("[Gemma] 원본 이미지만 입력")
            user_message["images"] = images
        else:
            print("[Gemma] 이미지 없음 → 텍스트 정보만 사용")

        response = _chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 금속 파단면 SEM 이미지에서 보이는 형상을 기록하는 관찰자다. "
                        "CNN 이 정한 파손 유형은 바꾸지 않는다. 한국어 JSON 만 출력한다."
                    ),
                },
                user_message,
            ],
            format=ANALYSIS_SCHEMA,
            options={
                "temperature": 0.25,
                "top_p": 0.9,
                "num_ctx": LLM_NUM_CTX,
                "num_predict": 1536,
            },
        )

        raw_response = response["message"]["content"].strip()
        print("Gemma 원본 응답:", raw_response)
        print("[Gemma] 종료 사유:", response.get("done_reason"))
        print("[Gemma] 생성 토큰 수:", response.get("eval_count"))

        parsed = parse_llm_json(raw_response)
        if not parsed:
            llm_error = "JSON 파싱 실패"
            print("JSON 파싱 실패")

    except Exception as e:
        llm_error = f"Gemma / Ollama 오류: {e}"
        print(llm_error)

    parsed = parsed or {}

    image_feature = validate_feature(prediction, parsed.get("feature", ""))
    observation_1 = validate_visual_observation(parsed.get("visual_observation_1", ""))
    observation_2 = validate_visual_observation(parsed.get("visual_observation_2", ""))
    mechanism_relation = validate_mechanism_relation(
        prediction, parsed.get("mechanism_relation", "")
    )
    llm_expected_cause = validate_expected_cause(
        prediction, parsed.get("expected_cause", ""), context["cause_candidates"]
    )
    cause_explanation = validate_cause_explanation(
        prediction, parsed.get("cause_explanation", "")
    )

    korean_explanation = compose_explanation(
        prediction,
        confidence_percent,
        observation_1,
        observation_2,
        mechanism_relation,
        cause_explanation,
    )

    print("[Gemma] 최종 주요 특징:", image_feature)
    print("[Gemma] 관찰 1:", observation_1)
    print("[Gemma] 관찰 2:", observation_2)
    print("[Gemma] 메커니즘:", mechanism_relation)
    print("[Gemma] 예상 원인:", llm_expected_cause)
    print("[Gemma] 원인 설명:", cause_explanation)

    return {
        "feature": image_feature,
        "visual_observation_1": observation_1,
        "visual_observation_2": observation_2,
        "mechanism_relation": mechanism_relation,
        "cause": llm_expected_cause,
        "expected_cause": llm_expected_cause,
        "cause_explanation": cause_explanation,
        "explanation": korean_explanation,
        "llm_error": llm_error,
    }


# =========================================================
# 분석 결과 비교
# =========================================================

COMPARE_TEXT_FIELDS = (
    "summary",
    "common_point",
    "visual_difference",
    "mechanism_difference",
    "cause_difference",
    "final_opinion",
)


def generate_compare_analysis(compare_items: list):

    def parse_confidence(value):
        if isinstance(value, str):
            value = value.replace("%", "").strip()
        try:
            return float(value)
        except Exception:
            return 0.0

    if len(compare_items) < 2:
        return {
            "summary": "비교를 위해서는 두 개 이상의 분석 결과가 필요합니다.",
            "common_point": "",
            "visual_difference": "",
            "mechanism_difference": "",
            "cause_difference": "",
            "confidence_difference": "",
            "final_opinion": "두 개 이상의 분석 결과를 선택해 주세요.",
        }

    item1 = compare_items[0]
    item2 = compare_items[1]

    # 기본 분석 정보
    pred1 = item1.get("prediction", "-")
    pred2 = item2.get("prediction", "-")

    conf1 = parse_confidence(item1.get("confidence", 0))
    conf2 = parse_confidence(item2.get("confidence", 0))

    feature1 = item1.get("feature", "-")
    feature2 = item2.get("feature", "-")

    obs1_1 = item1.get("visual_observation_1", "")
    obs1_2 = item1.get("visual_observation_2", "")
    obs2_1 = item2.get("visual_observation_1", "")
    obs2_2 = item2.get("visual_observation_2", "")

    mechanism1 = item1.get("mechanism_relation", "")
    mechanism2 = item2.get("mechanism_relation", "")

    cause1 = item1.get("expected_cause", "-")
    cause2 = item2.get("expected_cause", "-")

    cause_exp1 = item1.get("cause_explanation", "")
    cause_exp2 = item2.get("cause_explanation", "")

    # 기존 분석 기록과 호환
    if not obs1_1:
        obs1_1 = item1.get("explanation", "-")
    if not obs2_1:
        obs2_1 = item2.get("explanation", "-")

    # 신뢰도는 Python에서 직접 계산
    conf_gap = abs(conf1 - conf2)

    if conf_gap < 5:
        confidence_text = (
            f"분석 1은 {conf1:.1f}%, "
            f"분석 2는 {conf2:.1f}%로 "
            f"차이는 {conf_gap:.1f}%p이며 "
            "두 결과의 신뢰도는 유사한 수준입니다."
        )
    elif conf1 > conf2:
        confidence_text = (
            f"분석 1은 {conf1:.1f}%, "
            f"분석 2는 {conf2:.1f}%로 "
            f"분석 1이 {conf_gap:.1f}%p 높습니다."
        )
    else:
        confidence_text = (
            f"분석 1은 {conf1:.1f}%, "
            f"분석 2는 {conf2:.1f}%로 "
            f"분석 2가 {conf_gap:.1f}%p 높습니다."
        )

    if conf1 < 60 or conf2 < 60:
        confidence_text += (
            " 60% 미만의 결과가 포함되어 있어 "
            "해석 시 추가 검토가 필요할 수 있습니다."
        )

    # 동일 유형 / 다른 유형에 따른 비교 기준
    if pred1 == pred2:
        type_instruction = f"""
두 분석 모두 CNN이 "{pred1}"으로 예측하였다.
따라서 파손 유형 자체의 차이를 만들어내지 않는다.

같은 파손 유형 안에서
- 표면 형상
- 구조의 위치
- 크기
- 밀도
- 분포
- 방향성
- 영역별 차이
- 메커니즘 표현
- 예상 원인

이 어떻게 다른지를 중심으로 비교한다.
"""
    else:
        type_instruction = f"""
CNN 예측 결과가 서로 다르다.

분석 1: {pred1}
분석 2: {pred2}

두 파손 유형을 변경하거나 재분류하지 않는다.

각 이미지에서 실제 관찰된 특징을 기준으로
두 파손 유형에서 나타난 구조와
메커니즘 차이를 설명한다.
"""

    # 비교 Prompt
    prompt = f"""
너는 금속 파손 단면 분석 결과 두 개를
서로 비교하여 설명하는 도우미다.

중요:
너는 이미지를 다시 분류하지 않는다.
CNN이 결정한 파손 유형을 그대로 유지한다.

두 분석 결과에 실제로 존재하는 정보만 사용한다.


==================================================
[분석 1]
==================================================

파손 유형:
{pred1}

CNN 신뢰도:
{conf1:.1f}%

주요 특징:
{feature1}

관찰 특징 1:
{obs1_1}

관찰 특징 2:
{obs1_2}

메커니즘 관계:
{mechanism1}

예상 원인:
{cause1}

원인 설명:
{cause_exp1}


==================================================
[분석 2]
==================================================

파손 유형:
{pred2}

CNN 신뢰도:
{conf2:.1f}%

주요 특징:
{feature2}

관찰 특징 1:
{obs2_1}

관찰 특징 2:
{obs2_2}

메커니즘 관계:
{mechanism2}

예상 원인:
{cause2}

원인 설명:
{cause_exp2}


==================================================
[파손 유형 비교 조건]
==================================================

{type_instruction}


==================================================
[비교 우선순위]
==================================================

다음 순서로 비교한다.

1. 실제 관찰된 표면 형상
2. 구조의 위치
3. 구조의 크기
4. 구조의 밀도
5. 구조의 분포
6. 구조의 방향성
7. 영역별 차이
8. CNN 파손 메커니즘과의 관계
9. 예상 원인

입력 결과에 없는 특징은 추가하지 않는다.

차이가 명확하지 않은 항목은
억지로 차이를 만들지 않는다.


==================================================
[공통점]
==================================================

두 분석 결과에 실제로 공통적으로 나타난 특징만 설명한다.

파손 유형이 같다는 이유만으로
일반적 특징을 만들어내지 않는다.


==================================================
[시각적 차이]
==================================================

두 이미지의 관찰 결과를 직접 비교한다.

단순히 두 결과를 각각 나열하지 않는다.

가능하면

"분석 1은 ~인 반면,
분석 2는 ~"

형태로 직접 비교한다.


==================================================
[메커니즘 차이]
==================================================

입력된 mechanism_relation을 중심으로
각 이미지의 관찰 특징과
CNN 파손 메커니즘 관계의 차이를 설명한다.

여기에서는 파손 원인을 설명하지 않는다.


==================================================
[예상 원인 차이]
==================================================

각 분석에서 이미 제시된 expected_cause만 비교한다.

새로운 사고 원인을 생성하지 않는다.

두 원인이 동일하면 동일하다고 명확하게 설명한다.

예상 원인은 실제 사고 원인이 아니라
가능성을 나타내는 참고 정보로 설명한다.


==================================================
[신뢰도]
==================================================

신뢰도 계산 결과:

{confidence_text}

이 값을 변경하거나 다시 계산하지 않는다.

신뢰도를 이미지 품질이나
실제 파손 원인의 확실성과 동일시하지 않는다.


==================================================
[최종 해석]
==================================================

같은 유형이면
같은 파손 유형 안에서 어떤 관찰 차이가 있는지 설명한다.

다른 유형이면
각 파손 유형에서 어떤 관찰 차이가 나타나는지 설명한다.

이미지만으로 실제 사고 원인을 확정하지 않는다.


==================================================
[금지 사항]
==================================================

- CNN 결과 변경
- 새로운 파손 유형 제안
- 입력에 없는 이미지 특징 생성
- 실제 사고 상황 추측
- 부품 종류 추측
- 사용 환경 추측
- 재질 특성으로 차이 생성
- 원인 단정
- 동일 내용 반복


==================================================
[출력]
==================================================

반드시 한국어 JSON만 출력한다.

{{
    "summary": "두 분석의 가장 중요한 차이를 한 문장으로 요약",
    "common_point": "두 결과의 실제 공통점",
    "visual_difference": "두 이미지에서 관찰된 표면 특징의 직접적인 차이",
    "mechanism_difference": "관찰 특징과 파손 메커니즘 관계의 차이",
    "cause_difference": "두 분석에서 제시된 예상 원인의 차이",
    "final_opinion": "전체 비교에 대한 1~2문장 해석"
}}
"""

    empty_result = {key: "" for key in COMPARE_TEXT_FIELDS}
    empty_result["confidence_difference"] = confidence_text

    try:
        response = _chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 금속 파손 단면 분석 결과 두 개를 "
                        "비교하는 설명 생성기다. "
                        "CNN 결과를 변경하지 않는다. "
                        "각 이미지에서 이미 추출된 관찰 특징을 직접 비교한다. "
                        "두 결과를 따로 나열하지 말고 "
                        "공통점과 차이를 명확하게 제시한다. "
                        "입력에 없는 이미지 특징이나 사고 상황을 생성하지 않는다. "
                        "메커니즘과 원인을 구분하여 설명한다. "
                        "반드시 한국어 JSON만 출력한다."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.2, "top_p": 0.9, "num_ctx": LLM_NUM_CTX},
        )

        raw_response = response["message"]["content"].strip()
        print("Gemma 비교 원본 응답:", raw_response)

        parsed = parse_llm_json(raw_response)
        if not parsed:
            print("비교 JSON 파싱 실패")
            return dict(empty_result, llm_error="JSON 파싱 실패")

        result = {key: clean_text(parsed.get(key, "")) for key in COMPARE_TEXT_FIELDS}
        # 신뢰도는 LLM이 생성하지 않고 Python에서 계산한 문장을 그대로 사용
        result["confidence_difference"] = confidence_text
        result["llm_error"] = ""
        return result

    except Exception as e:
        print(f"Gemma 비교 분석 오류: {e}")
        return dict(empty_result, llm_error=f"Gemma 비교 분석 오류: {e}")


# =========================================================
# 보고서 보조 함수 (/report)
# =========================================================

def build_next_steps(prediction, prediction_en, confidence_percent, material, gradcam_summary,
                     extra_steps=None, background=None):
    # 입력 조건(촬영 위치·저배율·재질 불일치·배경 정보)에서 나온 항목을 맨 앞에 둔다.
    base = list(NEXT_STEPS[prediction])
    background = background or {}
    if background.get("environment"):
        base = [s for s in base if s["title"] != "사용 환경 확인"]
    if background.get("load") or background.get("temperature"):
        base = [s for s in base if s["title"] != "사용 온도와 충격 이력 확인"]
    steps = list(extra_steps or []) + base

    if material in UNKNOWN_MATERIALS:
        steps.append({
            "title": "재질 확인",
            "detail": "재질 정보가 없어 원인 해석에 반영하지 못했습니다. 성분 분석이나 경도 측정으로 재질을 확인하면 해석 정확도가 높아집니다.",
        })

    others = [
        KO_NAMES.get(k, k)
        for k in (gradcam_summary or {})
        if k != prediction_en
    ]
    if others:
        steps.append({
            "title": "함께 주목된 영역 확인",
            "detail": f"Grad-CAM++에서 {', '.join(others)} 관련 영역도 함께 주목되었습니다. 해당 부위를 확대 촬영해 별도로 관찰하세요.",
        })

    level = get_confidence_level(confidence_percent)
    if level == "low":
        steps.insert(0, {
            "title": "추가 이미지 촬영 및 전문가 검토",
            "detail": "예측 확률이 낮아 오분류 가능성이 있습니다. 다른 위치와 배율로 파단면을 추가 촬영하고 파손 해석 전문가의 검토를 받으세요.",
        })
    elif level == "medium":
        steps.append({
            "title": "추가 위치 촬영",
            "detail": "다른 위치와 배율의 이미지를 추가로 분석해 결과가 일관적인지 확인하세요.",
        })

    return steps


def build_confidence_comment(confidence_percent, note=""):
    comment = _confidence_comment(confidence_percent)
    return f"{comment} {note}".strip() if note else comment


def _confidence_comment(confidence_percent):
    level = get_confidence_level(confidence_percent)

    if level == "high":
        return (
            f"예측 확률은 {confidence_percent:.1f}%로 비교적 높은 편입니다. "
            "다만 이미지 분석만으로 실제 파손 원인을 확정할 수는 없으며, "
            "아래 확인 사항으로 검증하는 것이 좋습니다."
        )

    if level == "medium":
        return (
            f"예측 확률은 {confidence_percent:.1f}%로 중간 수준입니다. "
            "결과 해석에 주의가 필요하며 추가 이미지와 시험으로 보완하는 것이 좋습니다."
        )

    return (
        f"예측 확률은 {confidence_percent:.1f}%로 낮은 편입니다. "
        "오분류 가능성이 있으므로 추가 이미지와 전문가 검토가 필요합니다."
    )


def check_report_text(field, text, *, other_types=None,
                      need_uncertain=False, no_material=False):
    """규칙에 걸려도 문장을 바꾸지 않고 경고만 남긴다."""
    text = clean_text(text)

    if not text:
        warn(field, "빈 응답")
        return ""

    if any(w in text for w in FORBIDDEN_DEFINITIVE):
        warn(field, "단정 표현")

    if other_types and any(o in text for o in other_types):
        warn(field, "다른 유형명")

    if no_material and any(w in text for w in MATERIAL_WORDS):
        warn(field, "재질 미선택인데 재질 언급")

    if need_uncertain and not any(
        w in text for w in ["가능성", "추정", "수 있습니다", "보입니다"]
    ):
        warn(field, "추정형 아님")

    return ensure_sentence_end(text)


# =========================================================
# 보고서 Prompt
# =========================================================

def build_report_prompt(prediction, confidence_percent, material,
                        cam_text, feature, obs1, obs2, knowledge_text="",
                        imaging=None, must_text="", background=None):

    if material in UNKNOWN_MATERIALS:
        material_block = "재질 정보 없음. 재질이나 재질의 성질은 쓰지 않는다."
    else:
        material_ko = MATERIAL_LABELS.get(material, material)
        material_block = (
            f"사용자가 입력한 재질: {material_ko}\n"
            f"재질의 일반적 성질(참고): {MATERIAL_TRAITS.get(material, '')}\n"
            "cause 에서 이 성질을 이미지 관찰과 연결해 쓰되, 재질 성질만으로 원인을 단정하지 않는다."
        )

    imaging_text = imaging_info.prompt_text(imaging) or "- 입력 없음"
    background_text = background_info.prompt_text(background) or "- 입력 없음 (사용 환경·하중 조건을 지어내지 않는다)"

    must_block = must_text or "- 없음"

    knowledge_block = ""
    if knowledge_text:
        knowledge_block = (
            "\n[참고 지식 카드] (문헌 요약 일반 지식. 이 이미지에서 관찰한 사실이 아니다. "
            "[우선] 표시 카드를 먼저 반영한다.)\n"
            f"{knowledge_text}\n"
        )

    return f"""너는 금속 파손 단면 분석 결과를 보고서 문장으로 정리하는 도우미다.
파손 유형은 CNN 이 이미 정했으며 바꾸거나 다른 유형을 제안하지 않는다.

[CNN 결과]
- 최종 파손 유형: {prediction} (신뢰도 {confidence_percent:.1f}%)

[이 이미지의 관찰 결과] (이전 단계에서 이미지를 보고 기록한 내용)
- 주요 특징: {feature or "기록 없음"}
- 관찰 1: {obs1 or "기록 없음"}
- 관찰 2: {obs2 or "기록 없음"}

[모델 판단 근거]
{cam_text}

[재질 정보]
{material_block}

[SEM 촬영 정보]
{imaging_text}

[사전 배경 정보] (사용자가 입력한 부품·시험 조건. cause 에서 관찰과 연결해 쓴다)
{background_text}

[필수 반영 사항] 아래 각 항목의 의미가 괄호 안 필드에 반드시 드러나게 쓴다.
{must_block}
{knowledge_block}
[필드]
- characteristics (2~3문장): 먼저 [이 이미지의 관찰 결과]를 근거로 이 이미지에서 보이는 특징을 쓰고,
  그다음 그 특징이 {prediction}에서 갖는 일반적 의미를 1문장 덧붙인다.
- basis (2~3문장): Grad-CAM++ 강조 영역의 위치·분포와 확률을 근거로 이렇게 분류된 이유를 쓴다.
  강조 영역은 모델이 주목한 곳이지 실제 결함 위치가 아니다. 확률 수치는 나열하지 않는다.
- cause (3~4문장): 관찰과 입력 정보(재질·촬영 정보·사전 배경 정보)를 연결해 추정 원인을 쓴다. [필수 반영 사항]의 cause 항목을 포함한다.
  배경 정보와 관찰이 서로 맞지 않으면 맞지 않는다고 쓴다.
- next_steps_note (1문장): 정확한 원인 분석을 위해 추가 확인이 필요한 이유.
- used_cards: 위 필드에 내용을 실제로 쓴 카드 ID만 넣는다. 쓰지 않은 카드는 넣지 않는다.

[작성 규칙]
- 한국어로 쓴다. 원인은 "가능성이 있습니다", "추정할 수 있습니다"처럼 추정형으로 쓴다.
- "확실합니다", "분명합니다", "원인입니다", "틀림없습니다"는 쓰지 않는다.
- 입력에 없는 사용 환경·사고 상황은 만들지 않는다. 카드 내용을 이 이미지에서 관찰했다고 쓰지 않는다.
- 윤곽선 색이나 밝기를 특징으로 쓰지 않는다.

JSON 객체만 출력한다.
{{
    "characteristics": "...",
    "basis": "...",
    "cause": "...",
    "next_steps_note": "...",
    "used_cards": ["카드 ID"]
}}
"""


# =========================================================
# 보고서 생성
# =========================================================

def get_analysis_knowledge_refs(
    prediction,
    prediction_en,
    material="",
    probabilities=None,
    gradcam_summary=None,
    decision_path="",
    feature="",
    obs1="",
    obs2="",
    imaging=None,
    background=None,
):
    """분석 완료 시 보고서용 후보 지식 카드를 미리 조회한다."""

    imaging = imaging_info.normalize(imaging)
    background = background_info.normalize(background)

    context = report_rules.build_context(
        prediction_en,
        material,
        imaging,
        probabilities,
        background,
    )

    cam_text = build_cam_text(
        gradcam_summary,
        probabilities,
        decision_path,
    )

    cards = select_cards(
        prediction_en,
        material,
        obs_query=" ".join(
            str(x) for x in (prediction, feature, obs1, obs2) if x
        ),
        cam_query=cam_text,
        banned_words=(
            MATERIAL_WORDS if material in UNKNOWN_MATERIALS else ()
        ),
        imaging=imaging,
        background_cards=context["background_cards"],
    )

    return public_cards(cards, [])

def generate_report(prediction, prediction_en, confidence_percent,
                    material="", probabilities=None, gradcam_summary=None,
                    decision_path="", feature="", obs1="", obs2="", imaging=None,
                    background=None):

    no_material = material in UNKNOWN_MATERIALS
    imaging = imaging_info.normalize(imaging)
    other_types = [n for n in FRACTURE_TYPES if n != prediction]

    if prediction_en not in KO_NAMES:
        prediction_en = next(
            (en for en, ko in KO_NAMES.items() if ko == prediction), prediction_en
        )

    cam_text = build_cam_text(gradcam_summary, probabilities, decision_path)

    # 입력 조건 → 필수 반영 사항 (파이썬 규칙)
    background = background_info.normalize(background)
    context = report_rules.build_context(prediction_en, material, imaging, probabilities, background)
    must_text = report_rules.format_must_lines(context["must_lines"])
    if must_text:
        print("[필수 반영]\n" + must_text)

    cards = select_cards(
        prediction_en,
        material,
        obs_query=" ".join(x for x in (prediction, feature, obs1, obs2) if x),
        cam_query=cam_text,
        banned_words=MATERIAL_WORDS if no_material else (),
        imaging=imaging,
        background_cards=context["background_cards"],
    )

    print("[지식 카드] 선택:", ", ".join(c["id"] for c in cards) or "없음")

    prompt = build_report_prompt(
        prediction, confidence_percent, material,
        cam_text, feature, obs1, obs2,
        knowledge_text=format_cards_for_prompt(cards),
        imaging=imaging,
        must_text=must_text,
        background=background,
    )

    parsed = None
    llm_error = ""
    try:
        response = _chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 금속 파손 분석 보고서 작성 도우미다. "
                        "CNN 결과를 변경하지 않고 입력된 정보만 사용한다. "
                        "원인은 추정형으로 쓰고 반드시 한국어 JSON만 출력한다."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            format=REPORT_SCHEMA,
            options={"temperature": 0.25, "top_p": 0.9, "num_ctx": LLM_NUM_CTX,
                     "num_predict": 1536},
        )

        raw = response["message"]["content"].strip()
        print("Gemma 보고서 원본 응답:", raw)

        parsed = parse_llm_json(raw)
        if not parsed:
            llm_error = "보고서 JSON 파싱 실패"
            print(llm_error)

    except Exception as e:
        llm_error = f"보고서 Gemma 오류: {e}"
        print(llm_error)

    parsed = parsed or {}

    characteristics = check_report_text(
        "characteristics", parsed.get("characteristics", ""),
        other_types=other_types, no_material=no_material,
    )
    basis = check_report_text(
        "basis", parsed.get("basis", ""), no_material=no_material,
    )
    cause = check_report_text(
        "cause", parsed.get("cause", ""),
        other_types=other_types, need_uncertain=True,
        no_material=no_material,
    )
    note = check_report_text(
        "next_steps_note", parsed.get("next_steps_note", ""),
        no_material=no_material,
    )

    used_cards = parse_used_cards(parsed.get("used_cards", []), cards)
    used_cards = verify_used_cards(
        used_cards, cards, " ".join((characteristics, basis, cause, note))
    )

    return {
        "characteristics": characteristics,
        "basis": basis,
        "cause": cause,
        "next_steps_note": note,
        "next_steps": build_next_steps(
            prediction, prediction_en, confidence_percent,
            material, gradcam_summary, extra_steps=context["extra_steps"],
            background=background,
        ),
        "confidence_comment": build_confidence_comment(
            confidence_percent, context["reliability_note"]
        ),
        "confidence_status": get_confidence_level(confidence_percent),
        "material_label": (
            "모름 (보고서에 반영하지 않음)" if no_material
            else MATERIAL_LABELS.get(material, material)
        ),
        "imaging_label": imaging_info.label(imaging),
        "background_label": background_info.label(background),
        "knowledge_refs": public_cards(cards, used_cards),
        "llm_error": llm_error,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }