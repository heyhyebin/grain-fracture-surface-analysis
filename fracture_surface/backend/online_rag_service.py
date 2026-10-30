"""OpenAlex에서 MDPI 논문을 찾아 초록 기반 온라인 RAG 결과를 반환한다.

- 로컬 RAG와 독립적이며 오류 발생 시 빈 목록 반환
- 초록만 사용하므로 PDF 전문을 검색했다고 표시하지 않음
- OpenAlex 요청 및 필터링 통계를 출력하여 0건 원인 확인
"""
import json
import os
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

OPENALEX_URL = "https://api.openalex.org/works"
ONLINE_TIMEOUT = float(os.getenv("ONLINE_RAG_TIMEOUT", "12"))
SEARCH_TERMS = {
    "Cleavage": ("cleavage fracture", "brittle fracture fractography"),
    "Ductile": ("ductile fracture dimples", "microvoid coalescence"),
    "Fatigue": ("fatigue fracture striations", "fatigue crack fractography"),
    "Intergranular": ("intergranular fracture", "grain boundary embrittlement"),
}
MATCH_TERMS = {
    "Cleavage": ("cleavage", "brittle", "river pattern"),
    "Ductile": ("ductile", "dimple", "microvoid"),
    "Fatigue": ("fatigue", "striation", "cyclic"),
    "Intergranular": ("intergranular", "grain boundary", "embrittlement"),
}


def _abstract_from_index(index):
    if not isinstance(index, dict):
        return ""
    words = []
    for word, positions in index.items():
        if isinstance(positions, list):
            for position in positions:
                if isinstance(position, int) and 0 <= position < 10000:
                    words.append((position, word))
    words.sort()
    return " ".join(word for _, word in words)[:3000]


def _is_mdpi(work):
    """MDPI DOI 접두사 및 OpenAlex 저널/출판사 메타데이터로 판별."""
    doi = str(work.get("doi") or "").lower()
    # MDPI가 발행하는 학술 논문은 일반적으로 DOI 10.3390/을 사용한다.
    if "10.3390/" in doi:
        return True

    locations = [work.get("primary_location") or {}] + (work.get("locations") or [])
    for loc in locations:
        source = loc.get("source") or {}
        names = [
            source.get("host_organization_name"),
            source.get("display_name") if str(source.get("display_name") or "").lower().startswith("mdpi ") else "",
            *(source.get("host_organization_lineage_names") or []),
        ]
        if any("mdpi" in str(name).lower() for name in names if name):
            return True
        for url in (loc.get("landing_page_url"), loc.get("pdf_url")):
            host = (urlparse(str(url or "")).hostname or "").lower()
            if host == "mdpi.com" or host.endswith(".mdpi.com"):
                return True
    return False


def search_online_references(predicted_class_en, top_k=2):
    """MDPI 논문 초록을 검색해 기존 RAG 형식으로 반환한다. score는 None."""
    queries = SEARCH_TERMS.get(predicted_class_en, ())
    if not queries or top_k <= 0:
        return []

    found, seen = [], set()
    for query in queries:
        params = urlencode({"search": query, "filter": "is_oa:true", "per-page": 100})
        req = Request(OPENALEX_URL + "?" + params, headers={
            "User-Agent": "GrainFractographyResearch/1.0 (academic literature search)",
            "Accept": "application/json",
        })
        try:
            with urlopen(req, timeout=ONLINE_TIMEOUT) as response:
                payload = json.load(response)
        except Exception as exc:
            print(f"[온라인 RAG] '{query}' 요청 실패: {exc}")
            continue

        results = payload.get("results", [])
        mdpi_count = 0
        abstract_count = 0
        relevant_count = 0
        for work in results:
            if not isinstance(work, dict) or not _is_mdpi(work):
                continue
            mdpi_count += 1
            title = str(work.get("display_name") or "").strip()
            abstract = _abstract_from_index(work.get("abstract_inverted_index"))
            if not title or not abstract:
                continue
            abstract_count += 1
            if not any(term in (title + " " + abstract).lower()
                       for term in MATCH_TERMS[predicted_class_en]):
                continue
            relevant_count += 1
            doi = str(work.get("doi") or "")
            primary = work.get("primary_location") or {}
            url = doi or str(primary.get("landing_page_url") or "")
            if not url.startswith("https://"):
                continue
            identity = (doi or title).lower()
            if identity in seen:
                continue
            seen.add(identity)
            found.append({
                "source": title[:200],
                "page": None,
                "text": abstract,
                "score": None,
                "url": url,
                "source_type": "online_abstract",
                "year": work.get("publication_year"),
                "doi": doi,
            })
            if len(found) >= top_k:
                break

        print(f"[온라인 RAG] '{query}': 검색 {len(results)}건 / "
              f"MDPI {mdpi_count}건 / 초록 {abstract_count}건 / 관련 {relevant_count}건")
        if len(found) >= top_k:
            break

    print(f"[온라인 RAG] {predicted_class_en}: MDPI 초록 {len(found)}건")
    return found
