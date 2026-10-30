import { useState } from "react";

const SOURCE_INFO = {
  "metals-09-00148.pdf": {
    title: "Pantazopoulos (2019), Metals 9(2), 148",
    url: "https://www.mdpi.com/2075-4701/9/2/148",
  },
};

function KnowledgeItem({ card }) {
  const [expanded, setExpanded] = useState(false);
  const hasSources = Array.isArray(card.sources) && card.sources.length > 0;
  const hasText = Boolean(card.text);

  return (
    <li className="rounded-xl border border-emerald-100 bg-emerald-50/40 px-4 py-3">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="font-bold text-emerald-800">{card.id || "지식 카드"}</span>
        {card.section && <span className="text-slate-500">{card.section}</span>}
        {card.cited && (
          <span className="rounded bg-white px-2 py-0.5 text-emerald-700">
            LLM 인용 표시
          </span>
        )}
      </div>

      {hasText && (
        <p className={`mt-2 whitespace-pre-wrap break-words text-sm leading-6 text-slate-700 ${
          expanded ? "" : "line-clamp-2"
        }`}>
          {card.text}
        </p>
      )}

      {expanded && hasSources && (
        <div className="mt-2 space-y-1">
          {card.sources.map((s, j) => (
            <p key={j} className="break-words text-xs text-slate-500">
              {s.url ? (
                <a
                  href={s.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline hover:text-emerald-700"
                >
                  {s.citation || s.url}
                </a>
              ) : (
                s.citation
              )}
            </p>
          ))}
        </div>
      )}

      {(hasText || hasSources) && (
        <button
          type="button"
          onClick={() => setExpanded((prev) => !prev)}
          className="mt-2 text-xs font-semibold text-emerald-700 hover:underline"
          aria-expanded={expanded}
        >
          {expanded ? "접기" : "전체 설명 및 출처 보기"}
        </button>
      )}
    </li>
  );
}

function SourceItem({ item }) {
  const [expanded, setExpanded] = useState(false);
  const info = SOURCE_INFO[item.source] || {};
  const title = item.title || info.title || item.source || "출처 미상";
  const url = item.url || info.url;
  const snippet = String(item.snippet || "").trim();
  const isOnline = item.source_type === "online_abstract" || Boolean(item.doi);

  return (
    <li className="rounded-xl border border-slate-200 bg-white px-4 py-3">
      <div className="min-w-0">
        {url ? (
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm font-semibold text-slate-800 hover:text-emerald-700 hover:underline break-words"
          >
            {title}
          </a>
        ) : (
          <p className="text-sm font-semibold text-slate-800 break-words">{title}</p>
        )}
        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-500">
          <span>{isOnline ? "온라인 논문 초록" : "로컬 PDF"}</span>
          {!isOnline && item.page != null && item.page !== "" && <span>p.{item.page}</span>}
          {!isOnline && typeof item.score === "number" && (
            <span>검색 점수 {item.score.toFixed(2)}</span>
          )}
          {url && (
            <a
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="font-semibold text-emerald-700 hover:underline"
            >
              {isOnline ? "논문 보기" : "원문 보기"}
            </a>
          )}
          {snippet && (
            <button
              type="button"
              onClick={() => setExpanded((prev) => !prev)}
              className="font-semibold text-emerald-700 hover:underline"
              aria-expanded={expanded}
            >
              {expanded ? "초록/발췌 접기" : isOnline ? "초록 펼치기" : "발췌문 펼치기"}
            </button>
          )}
        </div>
      </div>
      {expanded && snippet && (
        <p className="mt-3 whitespace-pre-wrap break-words text-sm leading-6 text-slate-600">
          {snippet}
        </p>
      )}
    </li>
  );
}

export default function ReferenceCard({
  references = [],
  knowledgeRefs = [],
  reportLoading = false,
}) {
  const [showAllKnowledge, setShowAllKnowledge] = useState(false);
  const refs = Array.isArray(references) ? references : [];
  const cards = Array.isArray(knowledgeRefs) ? knowledgeRefs : [];
  const online = refs.filter((r) => r.source_type === "online_abstract" || Boolean(r.doi));
  const local = refs.filter((r) => r.source_type !== "online_abstract" && !r.doi);
  const visibleCards = showAllKnowledge ? cards : cards.slice(0, 3);

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-base font-bold text-slate-900">AI 설명 참고 자료</h3>
        <p className="mt-1 text-xs leading-5 text-slate-500">
          보고서 생성에 제공된 지식 카드와 파손 유형 기반 문헌 검색 결과를 구분해 표시합니다.
          참고 자료가 특정 이미지의 파손 원인을 입증하는 것은 아닙니다.
        </p>
      </div>

      <section>
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <h4 className="text-sm font-bold text-slate-800">파단면 분석 참고 지식</h4>
          {cards.length > 0 && (
            <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-xs text-emerald-700">
              {cards.length}건
            </span>
          )}
        </div>
        {cards.length > 0 ? (
          <>
            <ul className="space-y-2">
              {visibleCards.map((card, i) => (
                <KnowledgeItem key={`${card.id || "card"}-${i}`} card={card} />
              ))}
            </ul>
            {cards.length > 3 && (
              <button
                type="button"
                onClick={() => setShowAllKnowledge((prev) => !prev)}
                className="mt-3 w-full rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-2 text-sm font-semibold text-emerald-800 hover:bg-emerald-100"
                aria-expanded={showAllKnowledge}
              >
                {showAllKnowledge ? "참고 지식 접기" : `참고 지식 ${cards.length}건 모두 보기`}
              </button>
            )}
          </>
        ) : (
          <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 p-4 text-sm text-slate-500">
            {reportLoading
              ? "참고 지식을 불러오는 중입니다."
              : "이번 분석에서 선택된 참고 지식이 없습니다."}
          </div>
        )}
        {cards.some((c) => c.cited) && (
          <p className="mt-2 text-xs text-slate-500">
            ‘LLM 인용 표시’는 모델이 사용했다고 응답한 항목이며 실제 활용 여부를 검증한 표시는 아닙니다.
          </p>
        )}
      </section>

      {local.length > 0 && (
        <section>
          <h4 className="mb-2 text-sm font-bold text-slate-800">로컬 PDF 검색 자료</h4>
          <p className="mb-3 text-xs text-slate-500">
            예측된 파손 유형과 관련된 PDF 발췌문입니다.
          </p>
          <ul className="space-y-2">
            {local.map((item, i) => (
              <SourceItem key={`local-${i}`} item={item} />
            ))}
          </ul>
        </section>
      )}

      {online.length > 0 && (
        <section>
          <h4 className="mb-2 text-sm font-bold text-slate-800">온라인 관련 논문</h4>
          <p className="mb-3 text-xs leading-5 text-slate-500">
            예측된 파손 유형을 바탕으로 검색된 관련 연구입니다.
            업로드한 이미지와 직접 비교하거나 파손 원인을 검증한 결과는 아닙니다.
          </p>
          <ul className="space-y-2">
            {online.map((item, i) => (
              <SourceItem key={`online-${i}`} item={item} />
            ))}
          </ul>
        </section>
      )}

      {refs.length === 0 && cards.length === 0 && !reportLoading && (
        <p className="text-xs text-slate-400">
          현재 표시할 문헌 검색 결과가 없습니다.
        </p>
      )}
    </div>
  );
}
