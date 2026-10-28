"""
LLM 서비스 모듈 (Gemma / Ollama)

- generate_llm_analysis()     : 단일 이미지 멀티모달 분석 (/predict)
- generate_compare_analysis() : 두 분석 결과 비교
- generate_report()           : 파손단면 AI 분석 보고서 (/report)
"""

import re
import json
import ollama
from datetime import datetime

# =========================================================
# Ollama 모델 설정
# =========================================================

OLLAMA_MODEL = "gemma4:e2b"

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
    "unknown": "정보 없음",
    "": "미선택",
}

UNKNOWN_MATERIALS = ("", "unknown", None)

# =========================================================
# fallback용 기본 정보
# =========================================================

FALLBACK_FEATURES = {
    "취성 파괴": "표면의 균열 및 제한적인 변형 형태",
    "연성 파괴": "표면의 함몰 및 소성 변형 형태",
    "피로 파괴": "영역별 표면 차이 및 균열 성장 형태",
    "입계 파괴": "입자형 또는 경계형 분리 구조",
}

EXPECTED_CAUSE = {
    "취성 파괴": "급격한 응력 집중 가능성",
    "연성 파괴": "국부적인 하중 집중 가능성",
    "피로 파괴": "장기간 반복 하중 가능성",
    "입계 파괴": "결정립 경계 약화 가능성",
}

# 유형별 허용 원인 후보
CAUSE_CANDIDATES = {
    "취성 파괴": [
        "급격한 응력 집중 가능성",
        "충격성 하중의 영향 가능성",
        "국부적인 응력 집중에 따른 빠른 균열 진행 가능성",
    ],
    "연성 파괴": [
        "국부적인 하중 집중 가능성",
        "과도한 하중에 따른 변형 가능성",
        "국부적인 변형 집중 가능성",
        "과도한 변형이 파손에 영향을 주었을 가능성",
    ],
    "피로 파괴": [
        "장기간 반복 하중 가능성",
        "반복 응력에 따른 균열 성장 가능성",
        "국부적인 반복 응력이 균열 성장에 관여했을 가능성",
        "반복적인 하중에 따른 점진적 균열 전파 가능성",
    ],
    "입계 파괴": [
        "결정립 경계 약화 가능성",
        "결정립 경계 부근의 취약화 가능성",
        "입계 영역의 약화가 파손에 영향을 주었을 가능성",
    ],
}

# 메커니즘 fallback
FALLBACK_MECHANISM = {
    "취성 파괴": (
        "이러한 표면 형태는 CNN이 예측한 취성 파괴에서 "
        "제한적인 소성 변형과 함께 균열이 빠르게 진행되는 "
        "메커니즘과 관련해 해석할 수 있습니다."
    ),
    "연성 파괴": (
        "이러한 표면 형태는 CNN이 예측한 연성 파괴에서 "
        "소성 변형이 동반되는 파손 메커니즘과 "
        "관련해 해석할 수 있습니다."
    ),
    "피로 파괴": (
        "이러한 표면의 영역별 차이는 CNN이 예측한 피로 파괴에서 "
        "균열이 점진적으로 성장하는 메커니즘과 "
        "관련해 해석할 수 있습니다."
    ),
    "입계 파괴": (
        "이러한 표면 구조는 CNN이 예측한 입계 파괴에서 "
        "결정립 경계를 따라 분리가 진행되는 메커니즘과 "
        "관련해 해석할 수 있습니다."
    ),
}

# 원인 설명 fallback (보고서 cause 필드)
FALLBACK_CAUSE_EXPLANATION = {
    "취성 파괴": (
        "가능한 원인으로는 급격한 응력 집중이나 충격성 하중이 "
        "파손에 영향을 주었을 가능성이 있습니다."
    ),
    "연성 파괴": (
        "가능한 원인으로는 국부적인 하중 집중이나 과도한 변형이 "
        "파손에 영향을 주었을 가능성이 있습니다."
    ),
    "피로 파괴": (
        "가능한 원인으로는 장기간의 반복 하중이나 반복 응력이 "
        "균열 성장에 영향을 주었을 가능성이 있습니다."
    ),
    "입계 파괴": (
        "가능한 원인으로는 결정립 경계의 약화가 "
        "파손에 영향을 주었을 가능성이 있습니다."
    ),
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
}

REPORT_TYPE_INFO = {
    "취성 파괴": "거의 소성 변형 없이 균열이 빠르게 진행되어 결정면을 따라 쪼개지는 파괴이며, 평탄하고 반짝이는 분리면이 특징입니다.",
    "연성 파괴": "큰 소성 변형 후 파단되는 파괴이며, 미세한 함몰(딤플) 패턴과 컵-콘 형태의 변형 흔적이 특징입니다.",
    "피로 파괴": "반복 하중으로 균열이 서서히 성장하다 최종 파단되는 파괴이며, 비치 마크(줄무늬)와 구역별 표면 차이가 특징입니다.",
    "입계 파괴": "결정립 경계를 따라 균열이 전파되는 파괴이며, 입자 경계가 드러난 표면이 특징입니다.",
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
    "구리", "마그네슘", "니켈", "공구강", "강재", "탄소강",
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


def safe_visual_anchor(feature: str) -> str:
    """LLM 관찰 표현을 확정적 물리 구조명 대신 중립적 형상명으로 낮춘다.

    이미지 진위 판정은 불가능하므로 여기서 '검증 완료'라고 간주하지 않는다.
    """
    feature = clean_text(feature).strip().rstrip('.!?。')
    if not feature or len(feature) > 85:
        return ''
    if re.search(r'CNN|예측|원인|응력|하중|파괴|메커니즘|측정|입증', feature):
        return ''
    return feature.strip() if len(feature.strip()) >= 4 else ''


def conservative_cause_explanation(prediction: str, expected_cause: str) -> str:
    """선택 후보를 어색한 조사 중복 없이 보수적으로 표현한다."""
    candidate = clean_text(expected_cause).rstrip('.!?。 ')

    if candidate not in CAUSE_CANDIDATES[prediction]:
        candidate = EXPECTED_CAUSE[prediction]

    if candidate.endswith('가능성'):
        candidate = candidate[:-len('가능성')].strip()

    if '영향을 주었을' in candidate or '관여했을' in candidate:
        return ensure_sentence_end(f'가능한 원인으로 {candidate} 가능성이 있으나, 이미지만으로 확정할 수 없습니다')

    # '국부적인 하중 집중 등이'와 같이 명사형 원인을 사용한다.
    return ensure_sentence_end(f'가능한 원인으로 {candidate} 등이 고려될 수 있으나, 이미지만으로 확정할 수 없습니다')


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

def build_prompt(prediction: str, confidence_percent: float, material: str) -> str:
    """관찰을 우선하고, 해석·원인을 뒤에 분리하는 간결한 단일 호출 프롬프트."""
    condition = build_condition_text(material)
    confidence_note = get_confidence_instruction(confidence_percent)
    causes = "\n".join(f"- {item}" for item in CAUSE_CANDIDATES[prediction])

    return f"""너는 금속 파단면을 관찰하고 CNN 분석 결과를 설명하는 멀티모달 도우미다.

첫 이미지는 원본, 두 번째 이미지가 있다면 동일 이미지의 Grad-CAM++ 강조 시각화다.

Grad-CAM++은 CNN의 예측에 영향을 준 위치를 시각화할 뿐, 실제 응력·하중·균열 시작점을 측정하지 않는다.



[1단계: 순수 시각 관찰 — 가장 먼저 수행]

원본에서 눈에 보이는 형상(선형 무늬, 함몰, 판상 면, 경계, 거칠기)을 확인한다.

두 번째 이미지가 있다면 강조 위치와 그곳의 형상을 확인한다.

위치·분포·방향은 실제로 보일 때만 기술한다. 스케일 없는 절대 크기, 시간 경과, 균열 진행 방향은 추측하지 않는다.

금속 조각·섬유 구조·파편 등은 표면 형상의 시각적 묘사로 사용할 수 있으나, 실제 분리된 조각이나 재료 구조를 확인한 것으로 단정하지 않는다.

표면 무늬를 곧바로 파괴 원인이나 진행 과정으로 해석하지 않는다.

feature는 15~40자 정도의 짧은 명사구로 작성하고 원인·파손 유형·재질을 넣지 않는다.

visual_observation_1은 실제 보이는 대표 형상을, visual_observation_2는 관찰 가능한 위치·분포·방향을 기술한다.

관찰에 자신이 없으면 세부 형태를 만들어내지 말고 확실히 보이는 수준만 설명한다.
JSON 값이 없으면 실제 빈 문자열(\"\")을 사용하고, 빈 문자열이라는 단어를 출력하지 않는다.



[2단계: CNN 결과와 연결 — 관찰을 완료한 뒤 수행]

CNN 최종 예측: {prediction} (예측 확률 {confidence_percent:.1f}%). 예측 유형을 바꾸거나 재분류하지 않는다.

mechanism_relation에서는 관찰한 구체적 구조를 CNN의 {prediction} 예측과 연결한다.

연성 파괴: 소성 변형·미세 공극의 형성 및 결합.

취성 파괴: 제한적 소성 변형·균열을 따른 분리.

피로 파괴: 반복 하중과 연관될 수 있는 점진적 균열 성장 메커니즘. 단, 관찰만으로 반복 하중을 확인했다고 말하지 않는다.

입계 파괴: 결정립 경계를 따른 분리.

'뒷받침합니다', '입증합니다' 대신 '관련지어 해석할 수 있습니다'처럼 보수적으로 표현한다.

mechanism_relation에는 하중·응력 집중·경계 약화 등의 원인 추정을 넣지 않는다.



[3단계: 가능한 원인 — 마지막에 수행]

expected_cause는 다음 후보에서 하나를 정확히 선택한다. 이미지로 특정 원인을 판별할 수 없다면 첫 번째 기본 후보를 선택한다.

{causes}

cause_explanation에는 선택한 원인만 추정형으로 설명한다. 표면 구조가 하중이나 응력을 유발했다고 인과관계를 거꾸로 쓰지 않는다.

원인과 메커니즘은 이미지 하나로 확정할 수 없다.



참고 재질: {condition}

신뢰도에 따른 표현: {confidence_note}

반드시 한국어로, 필드당 한 문장 이내로, 아래 JSON 객체만 출력한다.

{{

  "feature": "실제 보이는 핵심 형상의 짧은 명사구",

  "visual_observation_1": "대표 형상 관찰 한 문장",

  "visual_observation_2": "위치·분포·방향 관찰 한 문장. 관찰할 내용이 없으면 값은 공백 없이 비워둔다",

  "mechanism_relation": "관찰 구조와 CNN 예측 메커니즘 연결 한 문장",

  "expected_cause": "위 허용 후보 하나",

  "cause_explanation": "선택한 원인의 보수적 추정 한 문장"

}}"""


def append_integrated_context(prompt, gradcam_summary=None, probabilities=None,
                              decision_path="", references_text=""):
    """Supplement original v7 prompt with RAG and structured CNN metadata."""
    parts = []
    if probabilities:
        parts.append("클래스별 CNN 출력 정보 (예측 재분류 금지): " + str(probabilities)[:1200])
    if gradcam_summary:
        parts.append("Grad-CAM++ 정량적 영역 요약 (위치 강조이며 실제 결함 확정 아님): " + str(gradcam_summary)[:1600])
    if decision_path:
        parts.append("CNN 판정 경로 (설명 참고용): " + str(decision_path)[:600])
    if references_text:
        parts.append("검색 문헌 참고 (이미지에 보이지 않는 특징을 관찰 사실로 주장하지 말 것):\n" + str(references_text)[:4500])
    return prompt + ("\n\n[추가 참고 정보]\n" + "\n".join(parts) if parts else "")


# =========================================================
# 단일 분석 응답 검증
# (모든 validate_* 함수는 빈 입력이면 해당 fallback 을 반환한다)
# =========================================================

def validate_feature(prediction: str, feature: str) -> str:
    """형상 서술만 보존. 시각적 진위는 정규식으로 판별할 수 없다."""
    feature = clean_text(feature).strip().rstrip('.!?。')

    if not feature:
        return FALLBACK_FEATURES[prediction]

    feature = re.sub(r'^(?:이미지에서|이미지에는|파단면에서|파단면에는)\s*', '', feature)

    # 해석절은 잘라내고 관찰절만 남긴다.
    feature = re.split(r'[,，]?\s*(?:이는|이러한 구조는|이러한 특징은)\s*(?=.{0,100}(?:파괴|메커니즘|응력|하중|관련))', feature, maxsplit=1)[0]

    if re.search(r'CNN|예측|응력|하중|원인|메커니즘|파괴|\d+\s*(?:mm|μm|cm)', feature):
        return FALLBACK_FEATURES[prediction]

    if re.search(r'붉은색|푸른색|빨간색|파란색|색상|밝기', feature):
        return FALLBACK_FEATURES[prediction]

    return feature if len(feature) >= 4 else FALLBACK_FEATURES[prediction]


def validate_visual_observation(text: str) -> str:
    """원인 해석만 잘라내고 가능한 관찰 절을 보존한다."""
    text = clean_text(text)
    if not text:
        return ''

    results = []
    for sentence in re.split(r'(?<=[.!?。])\s+', text):
        if re.search(r'\d+(?:\.\d+)?\s*(?:mm|cm|μm|µm|㎛)', sentence, re.I):
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


def validate_mechanism_relation(prediction: str, text: str, feature: str = '') -> str:
    """원인·과도한 단정을 제외하고 관찰 특징과 일반적 메커니즘을 연결한다."""
    text = clean_text(text)
    anchor = safe_visual_anchor(feature)

    mechanism = {
        '취성 파괴': '제한적인 소성 변형과 균열 진행',
        '연성 파괴': '소성 변형을 동반하는 파단 과정',
        '피로 파괴': '균열이 점진적으로 성장하는 과정',
        '입계 파괴': '결정립 경계를 따른 분리 과정',
    }[prediction]

    if not text or any(name != prediction and name in text for name in FRACTURE_TYPES):
        text = ''

    if re.search(r'확실|입증|증명|측정|반드시|응력 집중|하중 집중|반복(?:적인)? 하중|경계 약화|취약화|유발|가속화|촉진', text):
        text = ''

    terms = {
        '연성 파괴': ('소성 변형', '공극', '변형을 동반'),
        '취성 파괴': ('균열 전파', '균열 진행', '소성 변형', '분리'),
        '피로 파괴': ('균열 성장', '균열 전파', '점진적'),
        '입계 파괴': ('경계를 따라', '경계를 따른', '경계 분리', '입계 분리', '분리 과정'),
    }

    if text and any(term in text for term in terms[prediction]):
        return ensure_sentence_end(text)

    if anchor:
        return (f'이미지에서 묘사된 \\"{anchor}\\" 형태는 CNN의 {prediction} 예측에서 일반적으로 논의되는 '
                f'{mechanism}과 관련지어 해석할 수 있으나, 이 형상만으로 해당 과정을 확인할 수는 없습니다.')

    return FALLBACK_MECHANISM[prediction]


def validate_expected_cause(prediction: str, cause: str) -> str:
    cause = clean_text(cause)
    if not cause:
        return EXPECTED_CAUSE[prediction]

    for name in FRACTURE_TYPES:
        cause = cause.replace(name, "")
    cause = re.sub(r"\s+", " ", cause).strip()

    invalid_keywords = {
        "연성 파괴": ["반복 하중", "반복적인 하중", "반복 응력", "반복적인 응력", "피로", "결정립 경계"],
        "취성 파괴": ["반복 하중", "반복적인 하중", "반복 응력", "반복적인 응력", "피로", "큰 소성 변형"],
        "피로 파괴": ["단일 하중", "한 번의 큰 하중", "결정립 경계 약화", "과도한 소성 변형"],
        "입계 파괴": ["반복 하중", "반복적인 하중", "반복 응력", "반복적인 응력", "피로", "큰 소성 변형"],
    }
    for word in invalid_keywords.get(prediction, []):
        if word in cause:
            print(f"[cause validate] 부적절 원인 감지: {word}")
            return EXPECTED_CAUSE[prediction]

    candidate_keywords = {
        "취성 파괴": ["응력 집중", "충격"],
        "연성 파괴": ["하중 집중", "과도한 하중", "변형 집중", "과도한 변형"],
        "피로 파괴": ["반복 하중", "반복 응력", "균열 성장", "균열 전파"],
        "입계 파괴": ["결정립 경계", "입계", "취약화", "약화"],
    }
    if not any(word in cause for word in candidate_keywords[prediction]):
        print("[cause validate] 허용 원인 범위와 불일치 → fallback")
        return EXPECTED_CAUSE[prediction]

    return cause


def validate_cause_explanation(prediction: str, text: str, expected_cause: str = "") -> str:
    """추정형이더라도 특정 시각 구조의 형성/파괴를 직접 원인에 귀속하면 대체."""
    text = clean_text(text)
    fallback = conservative_cause_explanation(prediction, expected_cause)

    if not text or any(name != prediction and name in text for name in FRACTURE_TYPES):
        return fallback

    invalid_keywords = {
        '연성 파괴': ('반복 하중', '반복적인 하중', '반복 응력', '결정립 경계'),
        '취성 파괴': ('반복 하중', '반복적인 하중', '반복 응력', '큰 소성 변형'),
        '피로 파괴': ('한 번의 큰 하중', '급격한 단일 하중', '결정립 경계 약화'),
        '입계 파괴': ('반복 하중', '반복적인 하중', '반복 응력', '큰 소성 변형'),
    }
    if any(word in text for word in invalid_keywords[prediction]):
        return fallback

    surface = r'(?:구조|형태|무늬|균열|네트워크|틈새|분포|파편|공극|함몰|간격|층상)'
    cause = r'(?:응력|하중|충격|경계 약화|취약화)'

    # 표면 형상이 물리적 하중을 일으켰다는 역인과
    if re.search(surface + r'.{0,65}' + cause + r'.{0,25}(?:유발|발생시켰|만들었)', text):
        return fallback

    # 관찰 구조가 특정 하중 때문에 생겼거나 파괴됐다는 과도한 인과 귀속
    if re.search(surface + r'.{0,75}' + cause + r'.{0,30}(?:발생|형성|파괴|유발|생겨|만들)', text):
        return fallback
    if re.search(cause + r'.{0,70}' + surface + r'.{0,35}(?:발생|형성|파괴|유발|생겨|만들)', text):
        return fallback

    if not re.search(r'가능성|추정|고려될 수|해석될 수|영향을 미쳤을 수|작용했을 수', text):
        return fallback

    if re.search(r'확실|분명|입증|증명|측정|원인입니다', text):
        return fallback

    return ensure_sentence_end(
        text.replace('가능성이 매우 높습니다', '가능성이 있습니다')
            .replace('가능성이 높습니다', '가능성이 있습니다')
    )


def build_visual_sentences(prediction: str, feature: str, observation_1: str, observation_2: str):
    """관찰 문장 두 개 확보. 비어 있으면 feature 기반 문장/기본 문장으로 채운다."""
    observation_1 = validate_visual_observation(observation_1)
    observation_2 = validate_visual_observation(observation_2)

    if not observation_1:
        # feature가 완성 문장이면 그대로 사용해 '이미지에는 ... 형태가' 오류를 방지한다.
        feature = clean_text(feature).strip()
        if re.search(r"(?:습니다|입니다|됩니다|보입니다|있습니다)[.!?。]?$", feature):
            observation_1 = ensure_sentence_end(feature)
        elif feature:
            observation_1 = ensure_sentence_end(f"주요 표면 특징으로 {feature}가 관찰됩니다")
        else:
            observation_1 = "현재 이미지에서 구체적인 표면 특징을 확인하기 어렵습니다."

    if not observation_2 or observation_1 == observation_2:
        observation_2 = "추가적인 위치·분포 차이는 이미지에서 명확하게 확인하기 어렵습니다."

    return ensure_sentence_end(observation_1), ensure_sentence_end(observation_2)


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
):
    prompt = build_prompt(prediction, confidence_percent, material)
    prompt = append_integrated_context(
        prompt, gradcam_summary, probabilities, decision_path, references_text
    )

    parsed = None

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

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 금속 파손 단면 이미지의 시각적 특징을 "
                        "관찰하는 멀티모달 설명 생성기다. "
                        "파손 유형은 CNN이 이미 결정했으며 변경하지 않는다. "
                        "반드시 한국어 JSON만 출력한다."
                    ),
                },
                user_message,
            ],
            format="json",
            options={
                "temperature": 0.25,
                "top_p": 0.9,
                "num_ctx": 8192,
                "num_predict": 1024,
            },
        )

        raw_response = response["message"]["content"].strip()
        print("Gemma 원본 응답:", raw_response)
        print("[Gemma] 종료 사유:", response.get("done_reason"))
        print("[Gemma] 생성 토큰 수:", response.get("eval_count"))

        parsed = parse_llm_json(raw_response)
        if not parsed:
            print("JSON 파싱 실패 → fallback 사용")

    except Exception as e:
        print(f"Gemma / Ollama 오류: {e}")

    # LLM 응답이 없거나 실패해도 같은 검증 경로를 거친다.
    # 각 validate_* 가 빈 값에 대해 유형별 fallback 을 돌려준다.
    llm_ok = bool(parsed)
    parsed = parsed or {}

    image_feature = validate_feature(prediction, parsed.get("feature", ""))

    observation_1, observation_2 = build_visual_sentences(
        prediction,
        image_feature,
        parsed.get("visual_observation_1", ""),
        parsed.get("visual_observation_2", ""),
    )

    # LLM 이 실패했을 때는 fallback feature 를 근거로 문장을 만들지 않고
    # 기본 메커니즘 문장을 그대로 사용한다.
    mechanism_relation = validate_mechanism_relation(
        prediction,
        parsed.get("mechanism_relation", ""),
        image_feature if llm_ok else "",
    )

    llm_expected_cause = validate_expected_cause(
        prediction, parsed.get("expected_cause", "")
    )

    cause_explanation = validate_cause_explanation(
        prediction, parsed.get("cause_explanation", ""), llm_expected_cause
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

    fallback_result = {
        "summary":
            "두 분석 결과는 이미지에서 관찰된 표면 형태와 분포를 중심으로 비교할 수 있습니다.",
        "common_point":
            "두 결과 모두 이미지에서 관찰된 표면 특징을 바탕으로 CNN 결과를 해석하고 있습니다.",
        "visual_difference":
            "두 이미지의 세부적인 표면 구조와 분포를 함께 비교할 필요가 있습니다.",
        "mechanism_difference":
            "각 이미지에서 관찰된 특징과 CNN 파손 메커니즘의 관계를 비교할 수 있습니다.",
        "cause_difference":
            "예상 원인은 이미지 분석을 바탕으로 제시된 가능성이므로 참고 정보로 해석하는 것이 좋습니다.",
        "confidence_difference":
            confidence_text,
        "final_opinion":
            "두 결과의 관찰 특징과 CNN 분석을 함께 비교하여 해석하는 것이 좋으며, "
            "이미지만으로 실제 파손 원인을 확정하기는 어렵습니다.",
    }

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
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
            options={"temperature": 0.2, "top_p": 0.9},
        )

        raw_response = response["message"]["content"].strip()
        print("Gemma 비교 원본 응답:", raw_response)

        parsed = parse_llm_json(raw_response)
        if not parsed:
            return fallback_result

        result = {
            key: clean_text(parsed.get(key, "")) or fallback_result[key]
            for key in COMPARE_TEXT_FIELDS
        }
        # 신뢰도는 LLM이 생성하지 않고 Python에서 계산한 문장을 그대로 사용
        result["confidence_difference"] = confidence_text
        return result

    except Exception as e:
        print(f"Gemma 비교 분석 오류: {e}")
        return fallback_result


# =========================================================
# 보고서 보조 함수 (/report)
# =========================================================

def build_next_steps(prediction, prediction_en, confidence_percent, material, gradcam_summary):
    steps = list(NEXT_STEPS[prediction])

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


def build_confidence_comment(confidence_percent):
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


def fallback_basis(prediction, prediction_en, gradcam_summary):
    info = (gradcam_summary or {}).get(prediction_en)

    if not info:
        return (
            f"Grad-CAM++에서 뚜렷한 활성화 영역이 확인되지 않아 "
            f"모델의 전체 출력 확률을 기준으로 {prediction}로 분류되었습니다."
        )

    return (
        f"Grad-CAM++ 분석에서 {prediction}에 해당하는 활성화 영역이 "
        f"{info['position']}에 {info['spread']} 형태로 나타났으며, "
        f"모델은 이 영역을 판단 근거로 활용한 것으로 해석할 수 있습니다."
    )


def check_report_text(text, fallback, *, other_types=None,
                      need_uncertain=False, no_material=False):
    text = clean_text(text)

    if not text:
        return fallback

    if any(w in text for w in FORBIDDEN_DEFINITIVE):
        print("[report validate] 단정 표현 → fallback")
        return fallback

    if other_types and any(o in text for o in other_types):
        print("[report validate] 다른 유형명 → fallback")
        return fallback

    if no_material and any(w in text for w in MATERIAL_WORDS):
        print("[report validate] 재질 미선택인데 재질 언급 → fallback")
        return fallback

    if need_uncertain and not any(
        w in text for w in ["가능성", "추정", "수 있습니다", "보입니다"]
    ):
        print("[report validate] 추정형 아님 → fallback")
        return fallback

    return ensure_sentence_end(text)


# =========================================================
# 보고서 Prompt
# =========================================================

def build_report_prompt(prediction, confidence_percent, material,
                        cam_text, feature, obs1, obs2):

    if material in UNKNOWN_MATERIALS:
        material_block = (
            "재질 정보 없음.\n"
            "재질이나 재질의 성질을 절대 언급하지 않는다. "
            "이미지 관찰과 Grad-CAM++ 결과만으로 서술한다."
        )
    else:
        material_ko = MATERIAL_LABELS.get(material, material)
        material_block = (
            f"사용자가 입력한 재질: {material_ko}\n"
            f"재질의 일반적 성질(참고): {MATERIAL_TRAITS.get(material, '')}\n"
            "cause 필드에서 이 재질 성질을 이미지 관찰, Grad-CAM++ 결과와 "
            "연결해 설명한다. 재질 성질만으로 원인을 단정하지 않는다."
        )

    return f"""
너는 금속 파손 단면 분석 결과를 보고서 문장으로 정리하는 도우미다.
파손 유형은 CNN이 이미 결정했으며 변경하거나 다른 유형을 제안하지 않는다.

[CNN 결과]
- 최종 파손 유형: {prediction}
- 신뢰도: {confidence_percent:.1f}%

[이전 이미지 분석 결과]
- 주요 특징: {feature}
- 관찰 1: {obs1}
- 관찰 2: {obs2}

[모델 판단 근거]
{cam_text}

[재질 정보]
{material_block}

[작성 규칙]
- 반드시 한국어로 작성한다.
- 입력에 없는 특징, 사용 환경, 사고 상황을 만들어내지 않는다.
- Grad-CAM++ 영역은 모델이 주목한 영역이며 실제 결함 위치가 아니다.
- 원인은 반드시 추정형("가능성이 있습니다", "추정할 수 있습니다")으로 쓴다.
- "확실합니다", "분명합니다", "원인입니다", "틀림없습니다"를 쓰지 않는다.
- 색상과 밝기를 특징으로 쓰지 않는다.

[필드]
- characteristics: "{prediction}"의 일반적인 특징과 이 이미지에서 관찰된 특징 (2~3문장)
- basis: Grad-CAM++ 영역(위치, 분포, 겹침)과 확률을 근거로 이렇게 분류된 이유 (2~4문장). 확률 수치를 나열하지 않는다.
- cause: 추정되는 파손 원인 (3~4문장). 앞선 특징과 근거를 연결한다.
- next_steps_note: 정확한 원인 분석을 위해 아래 확인 사항이 필요한 이유 (1문장)

JSON 이외의 문장은 출력하지 않는다.

{{
    "characteristics": "...",
    "basis": "...",
    "cause": "...",
    "next_steps_note": "..."
}}
"""


# =========================================================
# 보고서 생성
# =========================================================

def generate_report(prediction, prediction_en, confidence_percent,
                    material="", probabilities=None, gradcam_summary=None,
                    decision_path="", feature="", obs1="", obs2=""):

    no_material = material in UNKNOWN_MATERIALS
    other_types = [n for n in FRACTURE_TYPES if n != prediction]

    fb_characteristics = REPORT_TYPE_INFO[prediction]
    fb_basis = fallback_basis(prediction, prediction_en, gradcam_summary)
    fb_cause = FALLBACK_CAUSE_EXPLANATION[prediction]
    if not no_material and material in MATERIAL_TRAITS:
        fb_cause += " 재질 특성도 해석 시 참고할 수 있습니다."
    fb_note = "이미지만으로는 원인을 확정할 수 없어 아래 항목을 추가로 확인하는 것이 필요합니다."

    cam_text = build_cam_text(gradcam_summary, probabilities, decision_path)
    prompt = build_report_prompt(
        prediction, confidence_percent, material,
        cam_text, feature, obs1, obs2,
    )

    parsed = None
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
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
            options={"temperature": 0.25, "top_p": 0.9},
        )

        raw = response["message"]["content"].strip()
        print("Gemma 보고서 원본 응답:", raw)

        parsed = parse_llm_json(raw)
        if not parsed:
            print("보고서 JSON 파싱 실패 → fallback")

    except Exception as e:
        print(f"보고서 Gemma 오류: {e}")

    # check_report_text 는 빈 값이면 fallback 을 돌려주므로
    # LLM 실패 시에도 같은 경로로 처리한다.
    parsed = parsed or {}

    characteristics = check_report_text(
        parsed.get("characteristics", ""), fb_characteristics,
        other_types=other_types, no_material=no_material,
    )
    basis = check_report_text(
        parsed.get("basis", ""), fb_basis, no_material=no_material,
    )
    cause = check_report_text(
        parsed.get("cause", ""), fb_cause,
        other_types=other_types, need_uncertain=True,
        no_material=no_material,
    )
    note = check_report_text(
        parsed.get("next_steps_note", ""), fb_note,
        no_material=no_material,
    )

    return {
        "characteristics": characteristics,
        "basis": basis,
        "cause": cause,
        "next_steps_note": note,
        "next_steps": build_next_steps(
            prediction, prediction_en, confidence_percent,
            material, gradcam_summary,
        ),
        "confidence_comment": build_confidence_comment(confidence_percent),
        "confidence_status": get_confidence_level(confidence_percent),
        "material_label": (
            "모름 (보고서에 반영하지 않음)" if no_material
            else MATERIAL_LABELS.get(material, material)
        ),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }