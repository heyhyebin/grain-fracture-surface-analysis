import ReferenceCard from "./components/ReferenceCard";
import {
  useState,
  useRef,
  useEffect,
  useMemo,
  useCallback,
} from "react";

import { createPortal } from "react-dom";

import {
  addHistoryItem,
  fetchHistory,
  clearAllHistory,
  saveReport,
  fetchReport,
  fetchReportIds,
  deleteHistoryItems,
  isDbStoreEnabled,
  setDbStoreEnabled,
  subscribeDbStore,
} from "./firebase";

import {
  buildPhaseHistoryItem,
  addPhaseHistoryItem,
  fetchPhaseHistory,
  renamePhaseHistoryItem,
  clearAllPhaseHistory,
  deletePhaseHistoryItems,
} from "./firebasePhase";

const CLASS_COLORS = {
  Cleavage: "#2563EB",
  Ductile: "#16A34A",
  Fatigue: "#D97706",
  Intergranular: "#DC2626",
};

const EN_NAMES = {
  "취성 파괴": "Cleavage",
  "연성 파괴": "Ductile",
  "피로 파괴": "Fatigue",
  "입계 파괴": "Intergranular",
};

// SEM 촬영 정보 (선택 입력, 비우면 "미지정")
const SEM_LOCATION_LABELS = {
  origin: "균열 시작점 부근",
  propagation: "균열 전파 영역",
  final: "최종 파단부",
  overview: "파면 전체 (저배율)",
};

const SEM_DETECTOR_LABELS = {
  SE: "이차전자 (SE)",
  BSE: "후방산란전자 (BSE)",
};

const YES_NO_LABELS = {
  yes: "함",
  no: "안 함",
};

const EMPTY_IMAGING = {
  magnification: "",
  scale_bar_um: "",
  location: "",
  detector: "",
  cleaned: "",
  coated: "",
};

// 백엔드가 돌려준 imaging(숫자·null 포함)을 입력란 값(문자열)으로 변환
const toImagingInputs = (imaging) => {
  const out = { ...EMPTY_IMAGING };
  Object.keys(out).forEach((k) => {
    const v = imaging?.[k];
    out[k] = v === null || v === undefined ? "" : String(v);
  });
  return out;
};

const imagingSummary = (imaging) => {
  const parts = [];
  if (imaging.magnification) parts.push(`${imaging.magnification}배`);
  if (imaging.scale_bar_um) parts.push(`스케일 바 ${imaging.scale_bar_um} µm`);
  if (imaging.location) parts.push(SEM_LOCATION_LABELS[imaging.location]);
  if (imaging.detector) parts.push(imaging.detector);
  if (imaging.cleaned) parts.push(`세척 ${YES_NO_LABELS[imaging.cleaned]}`);
  if (imaging.coated) parts.push(`코팅 ${YES_NO_LABELS[imaging.coated]}`);
  return parts.length ? parts.join(" · ") : "미지정";
};

// 사전 배경 정보 (선택 입력, 비우면 "미지정") — 값은 backend/background_info.py 와 같아야 한다
const BG_ENVIRONMENT_LABELS = {
  air: "일반 대기",
  chloride: "수분·염화물 (해수 등)",
  hydrogen: "수소 노출 (수소 충전·도금·산세·H₂S 등)",
  chemical: "산·알칼리·황화물 등 화학 약품",
};

const BG_LOAD_LABELS = {
  static: "정하중·단조 인장 (과부하·인장 시험)",
  impact: "충격",
  cyclic: "반복·진동",
  sustained: "장시간 지속 하중",
};

const BG_TEMPERATURE_LABELS = {
  low: "저온 (0 ℃ 이하·극저온)",
  room: "상온",
  high: "고온 (수백 ℃ 이상)",
};

const BG_PROCESS_LABELS = {
  heat_treated: "열처리",
  plated: "도금·산세",
  welded: "용접",
  cold_worked: "냉간 가공",
};

const BG_MANNER_LABELS = {
  sudden: "전조 없이 갑자기",
  gradual: "전조 있음 (균열 성장·소음·누설)",
};

const EMPTY_BACKGROUND = {
  environment: [],
  load: "",
  temperature: "",
  process: [],
  failure_manner: "",
  note: "",
};

// 백엔드가 돌려준 background 를 입력란 값으로 변환 (예전 기록에는 없을 수 있음)
const toBackgroundInputs = (bg) => ({
  environment: Array.isArray(bg?.environment) ? bg.environment : [],
  load: bg?.load || "",
  temperature: bg?.temperature || "",
  process: Array.isArray(bg?.process) ? bg.process : [],
  failure_manner: bg?.failure_manner || "",
  note: bg?.note || "",
});

const backgroundSummary = (bg, material) => {
  const parts = [];
  if (material && material !== "unknown") parts.push(MATERIAL_LABELS[material] || material);
  if (bg.environment.length)
    parts.push(bg.environment.map((e) => BG_ENVIRONMENT_LABELS[e].split(" (")[0]).join(", "));
  if (bg.load) parts.push(BG_LOAD_LABELS[bg.load].split(" (")[0]);
  if (bg.temperature) parts.push(BG_TEMPERATURE_LABELS[bg.temperature].split(" (")[0]);
  if (bg.process.length) parts.push(bg.process.map((p) => BG_PROCESS_LABELS[p]).join(", "));
  if (bg.failure_manner) parts.push(BG_MANNER_LABELS[bg.failure_manner].split(" (")[0]);
  if (bg.note) parts.push("메모 있음");
  return parts.length ? parts.join(" · ") : "미지정";
};

const MATERIAL_LABELS = {
  steel: "강 (Steel)",
  stainless_steel: "스테인리스강",
  aluminum: "알루미늄",
  titanium: "티타늄",
  cast_iron: "주철",
  copper: "구리",
  magnesium: "마그네슘 합금",
  nickel_alloy: "니켈 합금",
  tool_steel: "공구강",
  hea: "고엔트로피 합금 (HEA)",
  unknown: "미지정",
};

const PHASE_COLORS = {
  Al: "#997F4C",
  "Primary Si": "#4CB24C",
  Al3Ni: "#194C99",
  "Eutectic Si": "#CCCCCC",
};

const CONFIDENCE_BOX_STYLES = {
  high: "border-green-200 bg-green-50 text-green-700",
  medium: "border-amber-200 bg-amber-50 text-amber-700",
  low: "border-red-200 bg-red-50 text-red-700",
};

const layout = {
  page: "min-h-screen bg-[#f4f7fb] text-slate-900",
  container: "max-w-[1500px] mx-auto px-6",
};

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();

    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function decodeMaskFromImage(img, W, H) {
  const c = document.createElement("canvas");

  c.width = W;
  c.height = H;

  const ctx = c.getContext("2d");

  ctx.drawImage(img, 0, 0, W, H);

  const { data } = ctx.getImageData(0, 0, W, H);

  const mask = new Uint8Array(W * H);

  for (let i = 0; i < W * H; i++) {
    mask[i] = data[i * 4] > 127 ? 255 : 0;
  }

  return mask;
}

function maskToSegments(mask, W, H) {
  const segments = [];

  const get = (x, y) =>
    x < 0 || y < 0 || x >= W || y >= H
      ? 0
      : mask[y * W + x] > 0
      ? 1
      : 0;

  for (let y = -1; y < H; y++) {
    for (let x = -1; x < W; x++) {
      const tl = get(x, y);
      const tr = get(x + 1, y);
      const bl = get(x, y + 1);
      const br = get(x + 1, y + 1);

      const code =
        (tl << 3) |
        (tr << 2) |
        (br << 1) |
        bl;

      const top = {
        x: x + 1.0,
        y: y + 0.5,
      };

      const right = {
        x: x + 1.5,
        y: y + 1.0,
      };

      const bottom = {
        x: x + 1.0,
        y: y + 1.5,
      };

      const left = {
        x: x + 0.5,
        y: y + 1.0,
      };

      switch (code) {
        case 1:
          segments.push([left, bottom]);
          break;

        case 2:
          segments.push([bottom, right]);
          break;

        case 3:
          segments.push([left, right]);
          break;

        case 4:
          segments.push([top, right]);
          break;

        case 5:
          segments.push([left, top]);
          segments.push([bottom, right]);
          break;

        case 6:
          segments.push([top, bottom]);
          break;

        case 7:
          segments.push([left, top]);
          break;

        case 8:
          segments.push([top, left]);
          break;

        case 9:
          segments.push([top, bottom]);
          break;

        case 10:
          segments.push([top, right]);
          segments.push([left, bottom]);
          break;

        case 11:
          segments.push([top, right]);
          break;

        case 12:
          segments.push([left, right]);
          break;

        case 13:
          segments.push([bottom, right]);
          break;

        case 14:
          segments.push([left, bottom]);
          break;

        default:
          break;
      }
    }
  }

  return segments;
}

function segmentsToPolylines(segments) {
  const key = (p) =>
    `${p.x.toFixed(2)},${p.y.toFixed(2)}`;

  const map = new Map();

  segments.forEach((seg, i) => {
    const k1 = key(seg[0]);
    const k2 = key(seg[1]);

    if (!map.has(k1)) {
      map.set(k1, []);
    }

    if (!map.has(k2)) {
      map.set(k2, []);
    }

    map.get(k1).push(i);
    map.get(k2).push(i);
  });

  const used = new Array(
    segments.length
  ).fill(false);

  const polylines = [];

  for (
    let i = 0;
    i < segments.length;
    i++
  ) {
    if (used[i]) {
      continue;
    }

    used[i] = true;

    const poly = [
      segments[i][0],
      segments[i][1],
    ];

    let extended = true;

    while (extended) {
      extended = false;

      const tail =
        poly[poly.length - 1];

      const cands =
        map.get(key(tail)) || [];

      for (const ci of cands) {
        if (used[ci]) {
          continue;
        }

        const seg =
          segments[ci];

        if (
          key(seg[0]) ===
          key(tail)
        ) {
          poly.push(seg[1]);
          used[ci] = true;
          extended = true;
          break;
        }

        if (
          key(seg[1]) ===
          key(tail)
        ) {
          poly.push(seg[0]);
          used[ci] = true;
          extended = true;
          break;
        }
      }
    }

    polylines.push(poly);
  }

  return polylines;
}

function drawPolylines(
  ctx,
  polylines,
  color,
  {
    dashed = false,
    lineWidth = 3,
    dashPattern = null,
    dashOffset = 0,
  } = {}
) {
  ctx.save();

  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = "round";

  const isDashed =
    !!dashPattern || dashed;

  ctx.lineCap = isDashed
    ? "butt"
    : "round";

  if (dashPattern) {
    ctx.setLineDash(
      dashPattern
    );

    ctx.lineDashOffset =
      dashOffset;
  } else if (dashed) {
    ctx.setLineDash([
      16,
      10,
    ]);

    ctx.lineDashOffset = 0;
  } else {
    ctx.setLineDash([]);

    ctx.lineDashOffset = 0;
  }

  for (const poly of polylines) {
    if (poly.length < 2) {
      continue;
    }

    ctx.beginPath();

    ctx.moveTo(
      poly[0].x,
      poly[0].y
    );

    for (
      let i = 1;
      i < poly.length;
      i++
    ) {
      ctx.lineTo(
        poly[i].x,
        poly[i].y
      );
    }

    ctx.stroke();
  }

  ctx.restore();
}

// 좌표 JSON(gradcam_contours)을 캔버스에 그릴 수 있는 폴리라인으로 변환
function contoursToPolylines(contoursForClass) {
  return contoursForClass.map((c) => {
    const poly = c.pts.map(([x, y]) => ({ x, y }));
    if (poly.length > 1) {
      poly.push(poly[0]); // 닫힌 도형으로 만들기
    }
    return { poly, overlaps: c.overlaps_with || [] };
  });
}

// 보고서에 함께 저장할 Grad-CAM++ 스냅샷 (작은 JPEG data URL)
//  - 방금 분석한 결과: 서버가 준 gradcam_image 를 축소
//  - 기록에서 불러온 결과: base_image + gradcam_contours 로 윤곽선을 다시 그림
//  Firestore 문서 1MB 한도를 넘지 않도록 긴 변 maxSide px, JPEG 품질 quality 로 저장
async function makeGradcamSnapshot(result, maxSide = 800, quality = 0.8) {
  if (!result) return null;

  try {
    const hasContours =
      result.gradcam_contours &&
      Object.keys(result.gradcam_contours).length > 0;

    const src = result.gradcam_image || (hasContours ? result.base_image : null);
    if (!src) return null;

    const img = await loadImage(src);
    const W = img.naturalWidth;
    const H = img.naturalHeight;
    const k = Math.min(1, maxSide / Math.max(W, H));

    const canvas = document.createElement("canvas");
    canvas.width = Math.round(W * k);
    canvas.height = Math.round(H * k);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    if (!result.gradcam_image && hasContours) {
      ctx.save();
      ctx.scale(k, k);
      for (const [name, list] of Object.entries(result.gradcam_contours)) {
        const items = contoursToPolylines(list || []);
        const solo = items.filter((it) => it.overlaps.length === 0).map((it) => it.poly);
        const overlap = items.filter((it) => it.overlaps.length > 0).map((it) => it.poly);
        const color = CLASS_COLORS[name] || "#111827";
        if (solo.length) drawPolylines(ctx, solo, color, { lineWidth: 3 / k });
        if (overlap.length) {
          drawPolylines(ctx, overlap, color, {
            lineWidth: 4.5 / k,
            dashPattern: [16 / k, 10 / k],
          });
        }
      }
      ctx.restore();
    }

    return canvas.toDataURL("image/jpeg", quality);
  } catch (err) {
    console.error("Grad-CAM 스냅샷 생성 실패:", err);
    return null;
  }
}

function GradcamView({
  result,
  chipSize = "text-xs",
  canvasClass = "h-[320px]",
}) {
  const canvasRef =
    useRef(null);

  const baseImgRef =
    useRef(null);

  const masksRef =
    useRef({});

  const layerImgsRef =
    useRef({});

    const hasMasks =
    !!result.gradcam_masks;

  const hasLayers =
    !hasMasks && !!result.gradcam_layers;

  const hasContoursOnly =
    !hasMasks &&
    !hasLayers &&
    !!result.gradcam_contours &&
    Object.keys(result.gradcam_contours).length > 0;

  const sourceObj =
    useMemo(() => {
      if (hasMasks) return result.gradcam_masks;
      if (hasLayers) return result.gradcam_layers;
      return result.gradcam_contours || {};
    }, [
      hasMasks,
      hasLayers,
      result.gradcam_masks,
      result.gradcam_layers,
      result.gradcam_contours,
    ]);

  const allClasses =
    useMemo(() => {
      return Object.keys(
        sourceObj
      );
    }, [sourceObj]);

  const activeClasses =
    useMemo(() => {
      return allClasses.filter(
        (name) => {
          const contours =
            result
              .gradcam_contours?.[
              name
            ];

          return (
            contours &&
            contours.length > 0
          );
        }
      );
    }, [
      allClasses,
      result.gradcam_contours,
    ]);

  const [checked, setChecked] =
    useState(() =>
      Object.fromEntries(
        allClasses.map(
          (name) => [
            name,
            true,
          ]
        )
      )
    );

  const redraw =
    useCallback(() => {
      const canvas =
        canvasRef.current;

      const base =
        baseImgRef.current;

      if (
        !canvas ||
        !base
      ) {
        return;
      }

      const W =
        base.naturalWidth;

      const H =
        base.naturalHeight;

      canvas.width = W;
      canvas.height = H;

      const ctx =
        canvas.getContext("2d");

      ctx.clearRect(
        0,
        0,
        W,
        H
      );

      ctx.drawImage(
        base,
        0,
        0
      );

      if (hasMasks) {
        const active =
          allClasses.filter(
            (name) =>
              checked[name]
          );

        if (
          active.length === 0
        ) {
          return;
        }

        const masks = {};

        for (
          const name of active
        ) {
          if (
            masksRef.current[
              name
            ]
          ) {
            masks[name] =
              masksRef.current[
                name
              ];
          }
        }

        const DASH_ON = 14;
        const SLOT = 22;

        const isInsideMask = (
          x,
          y,
          name
        ) => {
          const m =
            masks[name];

          if (!m) {
            return false;
          }

          const ix =
            Math.round(x);

          const iy =
            Math.round(y);

          if (
            ix < 0 ||
            iy < 0 ||
            ix >= W ||
            iy >= H
          ) {
            return false;
          }

          return (
            m[
              iy * W +
                ix
            ] > 0
          );
        };

        const othersContaining =
          (
            p,
            selfName
          ) => {
            const found =
              [];

            for (
              const other of
              active
            ) {
              if (
                other ===
                selfName
              ) {
                continue;
              }

              if (
                isInsideMask(
                  p.x,
                  p.y,
                  other
                )
              ) {
                found.push(
                  other
                );
              }
            }

            return found;
          };

        for (
          const name of active
        ) {
          const mask =
            masks[name];

          if (!mask) {
            continue;
          }

          const layerCanvas =
            document.createElement(
              "canvas"
            );

          layerCanvas.width =
            W;

          layerCanvas.height =
            H;

          const layerCtx =
            layerCanvas.getContext(
              "2d"
            );

          const segs =
            maskToSegments(
              mask,
              W,
              H
            );

          if (
            segs.length ===
            0
          ) {
            ctx.drawImage(
              layerCanvas,
              0,
              0
            );

            continue;
          }

          const polylines =
            segmentsToPolylines(
              segs
            );

          for (
            const poly of
            polylines
          ) {
            if (
              poly.length < 2
            ) {
              continue;
            }

            const pointStates =
              poly.map(
                (p) => {
                  const others =
                    othersContaining(
                      p,
                      name
                    );

                  if (
                    others.length ===
                    0
                  ) {
                    return "solo";
                  }

                  return (
                    "overlap:" +
                    others
                      .sort()
                      .join(",")
                  );
                }
              );

            let segStart = 0;

            for (
              let i = 1;
              i <=
              pointStates.length;
              i++
            ) {
              const isEnd =
                i ===
                pointStates.length;

              const stateChanged =
                !isEnd &&
                pointStates[
                  i
                ] !==
                  pointStates[
                    segStart
                  ];

              if (
                isEnd ||
                stateChanged
              ) {
                const subPoly =
                  poly.slice(
                    segStart,
                    i +
                      (isEnd
                        ? 0
                        : 1)
                  );

                const state =
                  pointStates[
                    segStart
                  ];

                if (
                  state ===
                  "solo"
                ) {
                  drawPolylines(
                    layerCtx,
                    [
                      subPoly,
                    ],
                    CLASS_COLORS[
                      name
                    ],
                    {
                      lineWidth: 3,
                    }
                  );
                } else {
                  const others =
                    state
                      .slice(
                        "overlap:"
                          .length
                      )
                      .split(",");

                  const candidates =
                    [
                      name,
                      ...others,
                    ].sort();

                  const N =
                    candidates.length;

                  const myIndex =
                    candidates.indexOf(
                      name
                    );

                  const period =
                    N *
                    SLOT;

                  const dashPattern =
                    [
                      DASH_ON,
                      period -
                        DASH_ON,
                    ];

                  const dashOffset =
                    -myIndex *
                    SLOT;

                  drawPolylines(
                    layerCtx,
                    [
                      subPoly,
                    ],
                    CLASS_COLORS[
                      name
                    ],
                    {
                      lineWidth: 4.5,
                      dashPattern,
                      dashOffset,
                    }
                  );
                }

                segStart = i;
              }
            }
          }

          ctx.drawImage(
            layerCanvas,
            0,
            0
          );
        }
      } else if (hasContoursOnly) {
        // 윤곽선 전체에 부여된 overlaps_with 대신, 실제 겹치는 구간만 점선으로 그린다.
        const visible = allClasses.filter((name) => checked[name]);
        const shapes = Object.fromEntries(
          visible.map((name) => [
            name,
            contoursToPolylines(result.gradcam_contours?.[name] || []),
          ])
        );

        // 닫힌 윤곽선 안에 있는지 검사한다. 다른 유형의 영역 내부에 들어간
        // 선분만 겹침으로 간주하므로, 같은 윤곽선의 나머지 부분은 실선이 된다.
        const pointInPolygon = (p, poly) => {
          let inside = false;
          for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
            const a = poly[i];
            const b = poly[j];
            if ((a.y > p.y) !== (b.y > p.y) &&
                p.x < ((b.x - a.x) * (p.y - a.y)) / (b.y - a.y) + a.x) {
              inside = !inside;
            }
          }
          return inside;
        };

        const otherPolygons = Object.fromEntries(
          visible.map((name) => [
            name,
            visible
              .filter((other) => other !== name)
              .flatMap((other) => shapes[other].map((item) => item.poly))
              .filter((poly) => poly.length >= 4)
              .map((poly) => ({
                poly,
                minX: Math.min(...poly.map((p) => p.x)),
                maxX: Math.max(...poly.map((p) => p.x)),
                minY: Math.min(...poly.map((p) => p.y)),
                maxY: Math.max(...poly.map((p) => p.y)),
              })),
          ])
        );

        for (const name of visible) {
          const color = CLASS_COLORS[name] || "#111827";
          const polygons = otherPolygons[name];
          for (const item of shapes[name]) {
            const poly = item.poly;
            if (poly.length < 2) continue;

            // 각 선분의 중간점을 기준으로 겹침을 판정하고 연속 구간별로 그린다.
            // 한 유형만 켜져 있으면 polygons가 비어 있어 모두 실선이다.
            let run = [poly[0]];
            let runOverlap = null;
            const flush = () => {
              if (run.length < 2) return;
              drawPolylines(ctx, [run], color, runOverlap
                ? { dashed: true, lineWidth: 4.5 }
                : { lineWidth: 3 });
            };

            for (let i = 1; i < poly.length; i++) {
              const a = poly[i - 1];
              const b = poly[i];
              const midpoint = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
              const overlapping = polygons.some(({ poly: other, minX, maxX, minY, maxY }) =>
                midpoint.x >= minX && midpoint.x <= maxX &&
                midpoint.y >= minY && midpoint.y <= maxY &&
                pointInPolygon(midpoint, other)
              );
              if (runOverlap !== null && runOverlap !== overlapping) {
                flush();
                run = [a];
              }
              runOverlap = overlapping;
              run.push(b);
            }
            flush();
          }
        }
      } else {
        for (
          const name of
          allClasses
        ) {
          if (!checked[name]) continue;
          const layerImg = layerImgsRef.current[name];
          if (layerImg) {
            ctx.drawImage(layerImg, 0, 0);
          }
        }
      }
    }, [
      allClasses,
      checked,
      hasMasks,
      hasContoursOnly,
      result.gradcam_contours,
    ]);

  useEffect(() => {
    setChecked(
      Object.fromEntries(
        allClasses.map(
          (name) => [
            name,
            true,
          ]
        )
      )
    );
  }, [allClasses]);

  useEffect(() => {
    let cancelled =
      false;

    const loadAll =
      async () => {
        if (
          !result.base_image
        ) {
          return;
        }

        const base =
          await loadImage(
            result.base_image
          );

        if (cancelled) {
          return;
        }

        baseImgRef.current =
          base;

        const W =
          base.naturalWidth;

        const H =
          base.naturalHeight;

        if (hasMasks) {
          masksRef.current = {};
          for (const name of allClasses) {
            const src = sourceObj[name];
            if (!src) continue;
            const img = await loadImage(src);
            if (cancelled) return;
            masksRef.current[name] = decodeMaskFromImage(img, W, H);
          }
        } else if (hasContoursOnly) {
          // 좌표 데이터는 이미 result 안에 있어 별도 이미지 로딩이 필요 없음
        } else {
          layerImgsRef.current = {};
          for (const name of allClasses) {
            const src = sourceObj[name];
            if (!src) continue;
            layerImgsRef.current[name] = await loadImage(src);
            if (cancelled) return;
          }
        }

        redraw();
      };

    loadAll();

    return () => {
      cancelled = true;
    };
  }, [
    result.base_image,
    hasMasks,
    hasContoursOnly,  
    allClasses,
    sourceObj,
    redraw,
  ]);

  useEffect(() => {
    redraw();
  }, [redraw]);

  const toggle = (name) => {
    setChecked(
      (prev) => ({
        ...prev,
        [name]:
          !prev[name],
      })
    );
  };

  return (
    <div>
      <div
        className={`w-full ${canvasClass} rounded-xl border border-slate-200 bg-white overflow-hidden flex items-center justify-center`}
      >
        <canvas
          ref={
            canvasRef
          }
          className="w-full h-full object-contain"
          style={{
            display:
              "block",
          }}
        />
      </div>
      {activeClasses.length >
        0 && (
        <div className="flex flex-wrap justify-start gap-2 mt-3">
          {activeClasses.map(
            (name) => {
              const koName =
                Object.keys(
                  EN_NAMES
                ).find(
                  (key) =>
                    EN_NAMES[
                      key
                    ] ===
                    name
                );

              const color =
                CLASS_COLORS[
                  name
                ];

              const on =
                checked[
                  name
                ];

              return (
                <button
                  key={
                    name
                  }
                  onClick={() =>
                    toggle(
                      name
                    )
                  }
                  style={{
                    borderColor:
                      color,

                    backgroundColor:
                      on
                        ? color
                        : "transparent",

                    color:
                      on
                        ? "#fff"
                        : color,
                  }}
                  className={`px-3 py-1 rounded-full font-semibold border-2 transition ${chipSize}`}
                >
                  {koName ||
                    name}
                </button>
              );
            }
          )}
        </div>
      )}
    </div>
  );
}

// ==========================================
// DB 저장 On/Off 토글 (상단 바)
// ==========================================

function DbStoreToggle({ enabled, onChange }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={enabled}
      onClick={() => onChange(!enabled)}
      title={
        enabled
          ? "분석 기록을 데이터베이스(Firebase)에 저장하고 불러옵니다."
          : "데이터베이스를 사용하지 않습니다. 기록은 이 화면에만 임시로 남습니다."
      }
      className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/10 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:bg-white/15 transition"
    >
      <span className="whitespace-nowrap">DB 저장</span>
      <span
        className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition ${
          enabled ? "bg-emerald-500" : "bg-slate-500"
        }`}
      >
        <span
          className={`inline-block h-4 w-4 rounded-full bg-white shadow transition-transform ${
            enabled ? "translate-x-[18px]" : "translate-x-0.5"
          }`}
        />
      </span>
      <span className={`w-6 text-left ${enabled ? "text-emerald-300" : "text-slate-400"}`}>
        {enabled ? "ON" : "OFF"}
      </span>
    </button>
  );
}

// ==========================================
// 기록 선택 삭제 도구 막대 (사이드바)
// ==========================================

function DeleteToolbar({ selectedCount, totalCount, deleting, onToggleAll, onCancel, onDelete }) {
  const allSelected = totalCount > 0 && selectedCount === totalCount;

  return (
    <div className="mb-4 p-3 rounded-xl bg-red-50 border border-red-200">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs text-red-700 leading-5">
          삭제할 기록을 선택하세요.
          <span className="ml-1 font-semibold">({selectedCount}개 선택)</span>
        </p>

        <button
          type="button"
          onClick={onToggleAll}
          className="shrink-0 text-xs text-red-600 hover:text-red-800 font-semibold"
        >
          {allSelected ? "선택 해제" : "전체 선택"}
        </button>
      </div>

      <div className="mt-2 grid grid-cols-2 gap-2">
        <button
          type="button"
          onClick={onCancel}
          disabled={deleting}
          className="rounded-lg border border-slate-300 bg-white py-2 text-xs font-semibold text-slate-600 hover:bg-slate-50 disabled:opacity-40"
        >
          취소
        </button>

        <button
          type="button"
          onClick={onDelete}
          disabled={selectedCount === 0 || deleting}
          className="rounded-lg bg-red-600 py-2 text-xs font-semibold text-white hover:bg-red-700 disabled:opacity-30 disabled:cursor-not-allowed transition"
        >
          {deleting ? "삭제 중..." : `선택 삭제 (${selectedCount})`}
        </button>
      </div>
    </div>
  );
}

function ModalShell({
  title,
  subtitle,
  onClose,
  children,
  maxWidth = "max-w-5xl",
}) {
  return (
    <div className="fixed inset-0 z-[100] bg-slate-950/60 backdrop-blur-sm flex items-center justify-center p-5">
      <div
        className={`bg-white ${maxWidth} w-full max-h-[90vh] rounded-[24px] shadow-2xl overflow-hidden flex flex-col`}
      >
        <div className="flex items-start justify-between gap-5 px-7 py-5 border-b border-slate-200">
          <div>
            <h3 className="text-xl font-bold">
              {title}
            </h3>

            {subtitle && (
              <p className="text-sm text-slate-500 mt-1">
                {subtitle}
              </p>
            )}
          </div>
        <button
          onClick={onClose}
          className="w-9 h-9 rounded-full ..."
        >
          ×
        </button>
        </div>

        <div className="overflow-y-auto p-7">
          {children}
        </div>
      </div>
    </div>
  );
}

function ReportModal({ result, previewUrl, report, loading, error, onClose, saveState }) {
  // 기록에서 연 결과는 gradcam_image 가 비어 있으므로 저장된 스냅샷을 대신 표시
  const gradcamSrc = result.gradcam_image || report?.gradcam_snapshot || null;

  const probs = Object.entries(result.similarities || {}).sort((a, b) => {
    const va = parseFloat(a[1]);
    const vb = parseFloat(b[1]);
    return (isNaN(vb) ? -1 : vb) - (isNaN(va) ? -1 : va);
  });

  const statusStyle =
    CONFIDENCE_BOX_STYLES[report?.confidence_status] ||
    CONFIDENCE_BOX_STYLES.medium;

  const Section = ({ no, title, children }) => (
    <section className="mt-7 break-inside-avoid">
      <h3 className="flex items-center gap-2 text-base font-bold text-slate-900 border-b border-slate-300 pb-2">
        <span className="w-6 h-6 rounded-full bg-[#172536] text-white text-xs flex items-center justify-center">
          {no}
        </span>
        {title}
      </h3>
      <div className="mt-3">{children}</div>
    </section>
  );

  return createPortal(
    <div className="report-overlay fixed inset-0 z-[200] bg-slate-950/60 overflow-y-auto p-5">
      <style>{`
        @media print {
          #root { display: none !important; }
          .report-overlay {
            position: static !important;
            background: #fff !important;
            padding: 0 !important;
            overflow: visible !important;
          }
          .report-sheet {
            box-shadow: none !important;
            max-width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            border-radius: 0 !important;
          }
          .no-print { display: none !important; }
          .report-sheet, .report-sheet * {
            -webkit-print-color-adjust: exact;
            print-color-adjust: exact;
          }
        }
      `}</style>

      <div className="report-sheet bg-white max-w-4xl mx-auto rounded-2xl shadow-2xl p-9 text-slate-900">
        {/* 상단 버튼 */}
        <div className="no-print flex justify-end gap-2 mb-4">
          <button
            onClick={() => window.print()}
            disabled={!report}
            className="px-4 py-2 rounded-lg bg-[#172536] text-white text-sm font-semibold hover:bg-slate-700 disabled:opacity-40"
          >
            PDF 저장 / 인쇄
          </button>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-slate-100 text-slate-700 text-sm font-semibold hover:bg-slate-200"
          >
            닫기
          </button>
        </div>

        {/* 제목 */}
        <div className="border-b-2 border-[#172536] pb-4">
          <p className="text-xs tracking-[0.18em] uppercase text-blue-600 font-bold">
            Fracture Analysis Report
          </p>
          <div className="flex items-end justify-between gap-4 flex-wrap">
            <h2 className="text-2xl font-bold mt-1">
              파손단면 AI 분석 보고서
            </h2>

            <p className="text-xs text-slate-500">
              생성 시각: {report?.generated_at || "-"}
            </p>
          </div>

          <div className="flex flex-wrap gap-x-5 gap-y-1 mt-3 text-xs text-slate-500">
            <span>
              재질:{" "}
              {["모름", "unknown", "", null, undefined].includes(
                report?.material_label
              )
                ? "미지정"
                : report.material_label}
            </span>

            <span>
              SEM 촬영 정보:{" "}
              {["모름", "unknown", "", null, undefined].includes(
                report?.imaging_label
              )
                ? "미지정"
                : report.imaging_label}
            </span>

            <span>
              사전 배경 정보:{" "}
              {["모름", "unknown", "", null, undefined].includes(
                report?.background_label
              )
                ? "미지정"
                : report.background_label}
            </span>
          </div>

          {saveState?.status === "saving" && (
            <p className="no-print text-[11px] text-slate-400 mt-1">
              보고서 저장 중…
            </p>
          )}

          {saveState?.status === "saved" && (
            <p className="no-print text-[11px] text-emerald-600 mt-1">
              ✓ 분석 기록에 저장됨
              {saveState.at ? ` · ${new Date(saveState.at).toLocaleString()}` : ""}
            </p>
          )}
          {saveState?.status === "error" && (
            <p className="no-print text-[11px] text-red-500 mt-1">
              보고서 저장 실패: {saveState.message}
            </p>
          )}
        </div>

        {/* 1. 이미지 */}
        <Section no="1" title="입력 이미지 및 Grad-CAM++ 결과">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-slate-500 mb-1">입력 이미지</p>
              <div className="h-[260px] rounded-lg border border-slate-200 bg-slate-50 flex items-center justify-center overflow-hidden">
                {previewUrl && (
                  <img src={previewUrl} alt="입력 이미지" className="w-full h-full object-contain" />
                )}
              </div>
            </div>
            <div>
              <p className="text-xs text-slate-500 mb-1">Grad-CAM++ (모델 주목 영역)</p>
              <div className="h-[260px] rounded-lg border border-slate-200 bg-slate-50 flex items-center justify-center overflow-hidden">
                {gradcamSrc ? (
                  <img src={gradcamSrc} alt="Grad-CAM++" className="w-full h-full object-contain" />
                ) : (
                  <p className="text-xs text-slate-400">Grad-CAM++ 이미지 없음</p>
                )}
              </div>
            </div>
          </div>
        </Section>

        {/* 2. 분류 결과 */}
        <Section no="2" title="분류 결과">
          <div className="grid grid-cols-[0.7fr_1.3fr] gap-4 items-stretch">
            {/* 왼쪽: 최종 분류 */}
            <div className="rounded-xl bg-slate-50 border border-slate-200 p-5 flex flex-col justify-center">
              <p className="text-[11px] text-slate-500 uppercase tracking-[0.12em]">
                Final classification
              </p>
              <p className="text-xl font-black text-blue-700 mt-2 break-words">
                {result.display_prediction || result.prediction}
              </p>
              <p className="text-xs text-slate-500 mt-1">
                {result.prediction_en}
              </p>
              <p className="text-3xl font-black mt-3">
                {result.confidence}
              </p>
            </div>

            {/* 오른쪽: 유형별 막대그래프 */}
            <div className="rounded-xl border border-slate-200 p-4 flex flex-col justify-center gap-3">
              {probs.map(([name, value]) => {
                const percent = Math.max(
                  0,
                  Math.min(100, parseFloat(value) || 0)
                );
                const predictedName =
                  result.display_prediction || result.prediction || "";
                const isPred =
                  Boolean(predictedName) &&
                  (name === predictedName || name.includes(predictedName));
                return (
                  <div key={name} className="flex items-center gap-3">
                    {/* 왼쪽: 파손 유형 */}
                    <span
                      className={`w-[65px] shrink-0 text-[11px] ${
                        isPred ? "font-bold text-slate-900" : "text-slate-500"
                      }`}
                    >
                      {name}
                    </span>

                    {/* 가운데: 막대그래프 */}
                    <div className="flex-1 h-2 rounded-full bg-slate-100 overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          isPred ? "bg-blue-600" : "bg-slate-300"
                        }`}
                        style={{ width: `${percent}%` }}
                      />
                    </div>

                    {/* 오른쪽: 확률 */}
                    <span
                      className={`w-[40px] shrink-0 text-right text-[11px] ${
                        isPred ? "font-bold text-blue-700" : "text-slate-500"
                      }`}
                    >
                      {value}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
          <p className="text-[11px] text-slate-400 mt-2">
            "-"는 Grad-CAM++에서 검출되지 않은 유형입니다.
          </p>
        </Section>


        {loading && (
          <div className="mt-10 text-center py-10">
            <div className="w-9 h-9 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-sm text-slate-500">AI가 보고서를 작성하고 있습니다...</p>
          </div>
        )}

        {error && !loading && (
          <div className="mt-8 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-600">
            {error}
          </div>
        )}

        {report && !loading && (
          <>
            <Section no="3" title="파괴 유형의 특징">
              <p className="text-sm leading-7 text-slate-700">{report.characteristics}</p>
            </Section>

            <Section no="4" title="분류 근거 (Grad-CAM++ 기반)">
              <p className="text-sm leading-7 text-slate-700">{report.basis}</p>
            </Section>

            <Section no="5" title="추정 파손 원인">
              <p className="text-sm leading-7 text-slate-700">{report.cause}</p>
            </Section>

            <Section no="6" title="신뢰도 및 정확한 원인 분석을 위한 확인 사항">
              <div className={`rounded-lg border p-3 ${statusStyle}`}>
                <p className="text-sm leading-6">
                  {report.confidence_comment}
                </p>
              </div>

              {report.next_steps_note && (
                <p className="text-sm text-slate-700 leading-6 mt-3">
                  {report.next_steps_note}
                </p>
              )}

              <div className="grid grid-cols-2 gap-3 mt-3">
                {(Array.isArray(report.next_steps) ? report.next_steps : [])
                  .map((step, i) => (
                    <div
                      key={i}
                      className="rounded-xl border border-slate-200 bg-slate-50 p-3 break-inside-avoid"
                    >
                      <div className="flex items-start gap-2">
                        <span className="w-5 h-5 shrink-0 rounded-full bg-slate-200 text-slate-700 text-[11px] font-bold flex items-center justify-center">
                          {i + 1}
                        </span>
                        <p className="text-sm font-bold leading-5">
                          {step.title}
                        </p>
                      </div>

                      <p className="mt-2 text-xs text-slate-600 leading-5">
                        {step.detail}
                      </p>
                    </div>
                  ))}
              </div>
            </Section>

            {report.knowledge_refs?.length > 0 && (
              <Section no="7" title="참고 자료 및 출처">
                <p className="mb-3 text-xs text-slate-500">
                  보고서 작성에 제공된 참고 지식과 문헌 출처입니다.
                  실제 파손 원인을 입증하는 자료는 아닙니다.
                </p>

                <div className="rounded-lg border border-slate-200 overflow-hidden">
                  <div className="grid grid-cols-2 gap-3 bg-slate-100 px-3 py-2 text-xs font-bold">
                    <span>참고한 내용</span>
                    <span>자료 출처 / 링크</span>
                  </div>

                  {report.knowledge_refs.map((k, index) => {
                    const summary = String(k.text || k.section || "참고 내용 없음")
                      .split(/[.。\n]/)[0]
                      .trim();

                    const shortSummary =
                      summary.length > 90
                        ? `${summary.slice(0, 90)}…`
                        : summary;

                    return (
                      <div
                        key={`${k.id || "knowledge"}-${index}`}
                        className="grid grid-cols-2 gap-3 border-t border-slate-200 px-3 py-2 text-xs break-inside-avoid"
                      >
                        <div>
                          <div className="flex items-center gap-2 flex-wrap">
                            <p className="font-semibold text-slate-700">
                              {k.section || "참고 지식"}
                              {k.id && (
                                <span className="ml-1 text-slate-400">
                                  ({k.id})
                                </span>
                              )}
                            </p>

                            {k.cited === true && (
                              <span className="rounded-md bg-green-100 px-2 py-0.5 text-[10px] font-semibold text-green-700">
                                인용
                              </span>
                            )}
                          </div>

                          <p className="mt-1 text-slate-600 leading-5">
                            {shortSummary}
                          </p>
                        </div>

                        <div className="space-y-1">
                          {Array.isArray(k.sources) && k.sources.length > 0 ? (
                            k.sources.map((source, i) => (
                              <p key={i} className="text-slate-500 leading-5 break-words">
                                {source.url ? (
                                  <a
                                    href={source.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-blue-700 underline"
                                  >
                                    {source.citation || source.url}
                                  </a>
                                ) : (
                                  source.citation || "출처 정보 없음"
                                )}
                              </p>
                            ))
                          ) : (
                            <p className="text-slate-400">
                              출처 정보 없음
                            </p>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </Section>
            )}

            <p className="text-[11px] text-slate-400 leading-5 mt-8 border-t border-slate-200 pt-3">
              본 보고서는 이미지와 입력 정보를 기반으로 한 AI 추정 결과이며
              실제 파손 원인을 확정하는 자료가 아닙니다. 정확한 원인 규명을 위해서는
              전문가의 검토와 추가 시험이 필요합니다.
            </p>
          </>
        )}
      </div>
    </div>,
    document.body
  );
}

export default function App() {
  const fileRef =
    useRef(null);

  const [
    previewUrl,
    setPreviewUrl,
  ] = useState(null);

  const [
    thumbnailBase64,
    setThumbnailBase64,
  ] = useState(null);

  const [
    uploading,
    setUploading,
  ] = useState(false);

  const [
    material,
    setMaterial,
  ] = useState("unknown");

  const [imaging, setImaging] = useState(EMPTY_IMAGING);

  const updateImaging = (key) => (e) =>
    setImaging((prev) => ({ ...prev, [key]: e.target.value }));

  const [background, setBackground] = useState(EMPTY_BACKGROUND);

  const updateBackground = (key) => (e) =>
    setBackground((prev) => ({ ...prev, [key]: e.target.value }));

  // 복수 선택 항목 (사용 환경, 제조·처리 이력)
  const toggleBackground = (key, value) =>
    setBackground((prev) => {
      const list = prev[key].includes(value)
        ? prev[key].filter((v) => v !== value)
        : [...prev[key], value];
      return { ...prev, [key]: list };
    });

  const [
    result,
    setResult,
  ] = useState(null);

  const [
    history,
    setHistory,
  ] = useState([]);

  const [
    sidebarOpen,
    setSidebarOpen,
  ] = useState(true);

  const [
    selectedCompareIds,
    setSelectedCompareIds,
  ] = useState([]);

  const [
    showCompareModal,
    setShowCompareModal,
  ] = useState(false);

  const [
    compareSummary,
    setCompareSummary,
  ] = useState(null);

  const [
    compareLoading,
    setCompareLoading,
  ] = useState(false);

  const [
    showSimilarModal,
    setShowSimilarModal,
  ] = useState(false);

  const [
    showGradcamModal,
    setShowGradcamModal,
  ] = useState(false);

  const [showReport, setShowReport] = useState(false);
  const [report, setReport] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState("");

  // 보고서 ↔ 분석 기록 연결
  const [historyDocId, setHistoryDocId] = useState(null);   // 지금 보고 있는 결과의 기록 docId
  const historySaveRef = useRef(null);                       // 저장 중인 기록의 Promise<docId>
  const reportLoadReqRef = useRef(0);                        // 늦게 도착한 불러오기 응답 무시용
  const [reportSave, setReportSave] = useState(null);        // {status, at?, message?}
  const [reportIds, setReportIds] = useState(() => new Set()); // 보고서가 있는 기록 docId

  // DB 저장 On/Off (firebase.js 의 스위치와 동기화)
  const [dbEnabled, setDbEnabled] = useState(() => isDbStoreEnabled());
  useEffect(() => subscribeDbStore(setDbEnabled), []);

  // 기록 선택 삭제 모드 (현재 탭의 기록 목록에 적용)
  const [deleteMode, setDeleteMode] = useState(false);
  const [deleteIds, setDeleteIds] = useState([]);   // 선택한 기록의 id
  const [deleting, setDeleting] = useState(false);

  // 기록 목록이 바뀌면 어떤 기록에 보고서가 있는지 다시 조회 (목록 '보고서' 배지)
  useEffect(() => {
    if (!dbEnabled) return; // DB 를 끈 동안은 기존 배지 유지
    let alive = true;
    fetchReportIds(history.map((h) => h.docId))
      .then((ids) => {
        if (alive) setReportIds(ids);
      })
      .catch((err) => console.error("보고서 목록 조회 실패:", err));
    return () => {
      alive = false;
    };
  }, [history, dbEnabled]);

  const [activeTab, setActiveTab] = useState("fracture");

  const [detailTab, setDetailTab] = useState("rag"); // LLM 설명 탭은 보고서로 통합되어 제거

  const phaseFileRef = useRef(null);

  const [phasePreviewUrl, setPhasePreviewUrl] = useState(null);

  // ==========================================
  // Phase 분석 상태
  // ==========================================

  // 업로드한 원본 이미지 파일
  const [phaseFile, setPhaseFile] = useState(null);

  // Phase 분석 결과
  const [phaseResult, setPhaseResult] = useState(null);

  // Phase 분석 기록
  const [phaseHistory, setPhaseHistory] = useState([]);

  // Phase 분석 진행 상태
  const [phaseLoading, setPhaseLoading] = useState(false);

  // Phase 분석 오류 메시지
  const [phaseError, setPhaseError] = useState("");

// ==========================================
// Phase 이미지 업로드
// ==========================================

const handlePhaseFileChange = async (e) => {

  const file = e.target.files?.[0];

    if (!file) {
      return;
    }

    // 이미지 파일인지 확인
    if (!file.type.startsWith("image/")) {

      alert("이미지 파일만 업로드할 수 있습니다.");

      return;

    }

    try {

      // 이미지 미리보기 생성
      const base64 = await fileToBase64(file);

      // 실제 이미지 파일 저장
      setPhaseFile(file);

      // 이미지 미리보기 저장
      setPhasePreviewUrl(base64);

      // 이전 분석 결과 초기화
      setPhaseResult(null);

      // 이전 오류 초기화
      setPhaseError("");

    } catch (err) {

      console.error(
        "Phase 이미지 처리 실패:",
        err
      );

      alert(
        "이미지를 불러오는 중 오류가 발생했습니다."
      );

    }

  };

  // ==========================================
  // Phase 분석 실행
  // ==========================================

  const handlePhaseAnalyze = async () => {

    // 이미지가 없는 경우
    if (!phaseFile) {

      alert("Phase 분석용 이미지를 업로드해 주세요.");

      return;

    }

    // 중복 실행 방지
    if (phaseLoading) {
      return;
    }

    try {

      setPhaseLoading(true);

      setPhaseError("");

      setPhaseResult(null);

      // 서버로 전달할 데이터
      const formData = new FormData();

      formData.append(
        "file",
        phaseFile
      );

      // Phase 분석 API 호출
      const response = await fetch(
        "http://localhost:8000/analyze-phase",
        {
          method: "POST",
          body: formData,
        }
      );

      // HTTP 오류 처리
      if (!response.ok) {

        throw new Error(
          `Phase 분석 API 오류: ${response.status}`
        );

      }

      // 분석 결과 수신
      const data = await response.json();

if (data.status !== "success") {
  setPhaseError(data.message || "Phase 분석에 실패했습니다.");
  return;
}

setPhaseResult(data);

if (isDbStoreEnabled()) {
  // DB ON: 저장용 가벼운 기록 생성 후 Firestore 저장 → 목록 다시 불러오기
  try {
    const item = await buildPhaseHistoryItem({
      title: `Phase 분석 ${phaseHistory.length + 1}`,
      originalDataUrl: phasePreviewUrl,
      data,
    });
    await addPhaseHistoryItem(item);
    setPhaseHistory(await fetchPhaseHistory());
  } catch (err) {
    console.error("Phase 기록 저장 실패:", err);
    // 분석 결과는 이미 화면에 표시됐으므로 저장 실패만 알림
    alert("분석은 완료됐지만 기록 저장에 실패했습니다.");
  }
} else {
  // DB OFF: 화면 목록에만 임시로 추가 (docId 없음)
  setPhaseHistory((prev) => [
    {
      id: Date.now(),
      title: `Phase 분석 ${phaseHistory.length + 1}`,
      time: new Date().toLocaleString("ko-KR"),
      image: phasePreviewUrl,
      result: data,
      docId: null,
    },
    ...prev,
  ]);
}

      console.log(
        "Phase 분석 성공:",
        data
      );

    } catch (err) {

      console.error(
        "Phase 분석 실패:",
        err
      );

      setPhaseError(
        "Phase 분석 중 오류가 발생했습니다. 서버 연결 상태를 확인해 주세요."
      );

    } finally {

      setPhaseLoading(false);

    }

  };
 
  // ==========================================
  // Phase 분석 기록 클릭
  // ==========================================

  const handlePhaseHistoryClick = (item) => {
  if (!item) return;
  setPhasePreviewUrl(item.image);   // 썸네일(저해상도)
  setPhaseResult(item.result);
  setPhaseFile(null);               // 이전 파일이 재분석되는 문제 방지
  setPhaseError("");
};

  // ==========================================
  // Phase 기록 제목 변경
  // ==========================================
const handlePhaseHistoryRename = async (id, currentTitle) => {
  const newTitle = window.prompt("새로운 제목을 입력하세요.", currentTitle);
  if (newTitle === null) return;
  const trimmed = newTitle.trim();
  if (!trimmed) return;

  const target = phaseHistory.find((i) => i.id === id);
  setPhaseHistory((prev) =>
    prev.map((i) => (i.id === id ? { ...i, title: trimmed } : i))
  );
  try {
    if (target?.docId) await renamePhaseHistoryItem(target.docId, trimmed);
  } catch (err) {
    console.error("제목 변경 실패:", err);
  }
};

  useEffect(() => {
  const loadHistory = async () => {
    try {
      const items = await fetchHistory();
      setHistory(items);
    } catch (err) {
      console.error("Firestore 기록 불러오기 실패:", err);
    }
  };
  loadHistory();
}, []);

useEffect(() => {
  const loadPhaseHistory = async () => {
    try {
      setPhaseHistory(await fetchPhaseHistory());
    } catch (err) {
      console.error("Phase 기록 불러오기 실패:", err);
    }
  };
  loadPhaseHistory();
}, []);

  const materialText =
    MATERIAL_LABELS[result?.material] ||
    (result?.material && result.material !== "미지정"
      ? result.material
      : "미지정");

  const confidenceStyle =
    CONFIDENCE_BOX_STYLES[
      result
        ?.confidence_status
    ] ||
    CONFIDENCE_BOX_STYLES
      .medium;

  const confidenceLabel =
    result
      ?.confidence_status ===
    "high"
      ? "높음"
      : result
          ?.confidence_status ===
        "low"
      ? "낮음"
      : "보통";

  const compareItems =
    history.filter(
      (item) =>
        selectedCompareIds.includes(
          item.id
        )
    );

  const sameCompareCause =
    compareItems.length >=
      2 &&
    (
      compareItems[0]
        ?.result
        ?.expected_cause ||
      ""
    ).trim() ===
      (
        compareItems[1]
          ?.result
          ?.expected_cause ||
        ""
      ).trim();

  const fileToBase64 = (
    file
  ) => {
    return new Promise(
      (
        resolve,
        reject
      ) => {
        const reader =
          new FileReader();

        reader.onload =
          () =>
            resolve(
              reader.result
            );

        reader.onerror =
          reject;

        reader.readAsDataURL(
          file
        );
      }
    );
  };

  const makeThumbnail = (
    file,
    maxSize = 180,
    quality = 0.65
  ) => {
    return new Promise(
      (
        resolve,
        reject
      ) => {
        const reader =
          new FileReader();

        const img =
          new Image();

        reader.onload =
          () => {
            img.onload =
              () => {
                const canvas =
                  document.createElement(
                    "canvas"
                  );

                const scale =
                  Math.min(
                    maxSize /
                      img.width,
                    maxSize /
                      img.height
                  );

                canvas.width =
                  Math.round(
                    img.width *
                      scale
                  );

                canvas.height =
                  Math.round(
                    img.height *
                      scale
                  );

                const ctx =
                  canvas.getContext(
                    "2d"
                  );

                ctx.drawImage(
                  img,
                  0,
                  0,
                  canvas.width,
                  canvas.height
                );

                resolve(
                  canvas.toDataURL(
                    "image/jpeg",
                    quality
                  )
                );
              };

            img.onerror =
              reject;

            img.src =
              reader.result;
          };

        reader.onerror =
          reject;

        reader.readAsDataURL(
          file
        );
      }
    );
  };

  const handleFileChange =
    async (e) => {
      const file =
        e.target.files[0];

      if (!file) {
        return;
      }

      try {
        const base64 =
          await fileToBase64(
            file
          );

        const thumbnail =
          await makeThumbnail(
            file
          );

        setPreviewUrl(
          base64
        );

        setThumbnailBase64(
          thumbnail
        );

        setResult(null);
        setReport(null);
        setShowReport(false);
        setReportError("");
        setReportSave(null);
        setHistoryDocId(null);
        historySaveRef.current = null;
        reportLoadReqRef.current++;

        setShowSimilarModal(
          false
        );

        setShowGradcamModal(
          false
        );
      } catch (err) {
        console.error(
          "이미지 처리 실패:",
          err
        );

        alert(
          "이미지를 불러오는 중 오류가 발생했습니다."
        );
      }
    };

  const saveHistory = async (data, thumbnail) => {
  const hasNewMasks = !!data.gradcam_masks;
  const historyResult = {
    ...data,
    gradcam_image: null,
    gradcam_masks: null,      // Firestore 1MB 제한 고려, 마스크도 제외
    gradcam_layers: null,
  };

  const newItem = {
    id: Date.now(),
    time: new Date().toLocaleString(),
    image: thumbnail,
    result: historyResult,
  };

  // DB OFF: 화면 목록에만 임시로 추가 (docId 없음)
  if (!isDbStoreEnabled()) {
    setHistoryDocId(null);
    setHistory((prev) => [{ ...newItem, docId: null }, ...prev]);
    return null;
  }

  try {
    const docId = await addHistoryItem(newItem);
    setHistoryDocId(docId);
    const items = await fetchHistory();
    setHistory(items);
    return docId;
  } catch (err) {
    console.error("Firestore 기록 저장 실패:", err);
    alert("기록 저장 중 오류가 발생했습니다. 콘솔을 확인해주세요.");
    return null;
  }
};

  const handleHistoryClick =
    (item) => {
      setResult(
        item.result
      );

      setReport(null);
      setShowReport(false);
      setReportError("");
      setReportSave(null);
      setHistoryDocId(item.docId || null);
      historySaveRef.current = null;
      loadSavedReport(item.docId);

      setPreviewUrl(
        item.image
      );

      setThumbnailBase64(
        item.image
      );

      setMaterial(
        item.result.material || "unknown"
      );

      setImaging(toImagingInputs(item.result.imaging));
      setBackground(toBackgroundInputs(item.result.background));

      setShowSimilarModal(
        false
      );

      setShowGradcamModal(
        false
      );
    };

  const toggleCompareSelect =
    (id) => {
      setSelectedCompareIds(
        (prev) => {
          if (
            prev.includes(
              id
            )
          ) {
            return prev.filter(
              (itemId) =>
                itemId !==
                id
            );
          }

          if (
            prev.length >=
            2
          ) {
            alert(
              "비교는 최대 2개까지 선택할 수 있습니다."
            );

            return prev;
          }

          return [
            ...prev,
            id,
          ];
        }
      );
    };

const clearHistory = async () => {
  try {
    await clearAllHistory();
  } catch (err) {
    console.error("Firestore 기록 삭제 실패:", err);
  }
  setHistory([]);
  setReportIds(new Set());
  setHistoryDocId(null);
  setReportSave(null);
  setSelectedCompareIds([]);
  setShowCompareModal(false);
  setCompareSummary(null);
};

// ==========================================
// 기록 선택 삭제
// ==========================================

const exitDeleteMode = () => {
  setDeleteMode(false);
  setDeleteIds([]);
};

// 탭을 바꾸면 선택 삭제 모드 해제
useEffect(() => {
  setDeleteMode(false);
  setDeleteIds([]);
}, [activeTab]);

const toggleDeleteSelect = (id) => {
  setDeleteIds((prev) =>
    prev.includes(id) ? prev.filter((itemId) => itemId !== id) : [...prev, id]
  );
};

const toggleDeleteSelectAll = () => {
  const list = activeTab === "fracture" ? history : phaseHistory;
  setDeleteIds((prev) =>
    prev.length === list.length ? [] : list.map((item) => item.id)
  );
};

const reloadHistoryFromDb = async () => {
  try {
    setHistory(await fetchHistory());
  } catch (err) {
    console.error("Firestore 기록 불러오기 실패:", err);
  }
  try {
    setPhaseHistory(await fetchPhaseHistory());
  } catch (err) {
    console.error("Phase 기록 불러오기 실패:", err);
  }
};

const handleDeleteSelected = async () => {
  const isFracture = activeTab === "fracture";
  const list = isFracture ? history : phaseHistory;
  const targets = list.filter((item) => deleteIds.includes(item.id));
  if (targets.length === 0) return;

  const dbOn = isDbStoreEnabled();
  const docIds = targets.map((item) => item.docId).filter(Boolean);

  const message =
    `선택한 기록 ${targets.length}개를 삭제할까요?\n` +
    (dbOn
      ? "데이터베이스에서도 삭제되며 되돌릴 수 없습니다."
      : docIds.length > 0
        ? "DB 저장이 꺼져 있어 화면 목록에서만 지워집니다. (DB를 다시 켜면 다시 나타납니다)"
        : "화면 목록에서 지워집니다.");

  if (!window.confirm(message)) return;

  setDeleting(true);

  try {
    if (dbOn && docIds.length > 0) {
      if (isFracture) await deleteHistoryItems(docIds);
      else await deletePhaseHistoryItems(docIds);
    }
  } catch (err) {
    console.error("기록 선택 삭제 실패:", err);
    alert("기록 삭제 중 오류가 발생했습니다. 콘솔을 확인해주세요.");
    await reloadHistoryFromDb(); // 일부만 지워졌을 수 있으므로 DB 기준으로 다시 표시
    setDeleting(false);
    return;
  }

  const removed = new Set(targets.map((item) => item.id));

  if (isFracture) {
    setHistory((prev) => prev.filter((item) => !removed.has(item.id)));
    setSelectedCompareIds((prev) => prev.filter((id) => !removed.has(id)));
    setReportIds((prev) => {
      const next = new Set(prev);
      docIds.forEach((id) => next.delete(id));
      return next;
    });
    if (historyDocId && docIds.includes(historyDocId)) {
      // 지금 보고 있는 결과의 기록이 지워짐 → 보고서 연결 해제
      setHistoryDocId(null);
      setReportSave(null);
    }
    setCompareSummary(null);
  } else {
    setPhaseHistory((prev) => prev.filter((item) => !removed.has(item.id)));
  }

  setDeleting(false);
  exitDeleteMode();
};

// ==========================================
// DB 저장 On/Off
// ==========================================

const handleDbToggle = async (next) => {
  if (next) {
    const hasLocalOnly =
      history.some((item) => !item.docId) ||
      phaseHistory.some((item) => !item.docId);

    if (
      hasLocalOnly &&
      !window.confirm(
        "DB 저장을 켜면 데이터베이스의 기록을 다시 불러옵니다.\n" +
          "DB를 끈 동안 분석한 기록은 저장되지 않았으므로 목록에서 사라집니다. 계속할까요?"
      )
    ) {
      return;
    }

    setDbStoreEnabled(true);
    exitDeleteMode();
    setSelectedCompareIds([]);
    setCompareSummary(null);
    await reloadHistoryFromDb();
  } else {
    // 끌 때는 지금 보이는 목록을 그대로 두고, 이후 분석은 화면에만 기록
    setDbStoreEnabled(false);
  }
};

  const handleUpload =
    async () => {
      const file =
        fileRef.current
          ?.files[0];

      if (!file) {
        return alert(
          "이미지를 먼저 선택해주세요."
        );
      }

      const formData =
        new FormData();

      formData.append(
        "file",
        file
      );

      formData.append(
        "material",
        material
      );

      // SEM 촬영 정보 (빈 값 = 미지정)
      Object.entries(imaging).forEach(([key, value]) => {
        formData.append(`sem_${key}`, value);
      });

      // 사전 배경 정보 (빈 값 = 미지정, 복수 선택은 쉼표로 구분)
      Object.entries(background).forEach(([key, value]) => {
        formData.append(`bg_${key}`, Array.isArray(value) ? value.join(",") : value);
      });

      try {
        setUploading(true);
        setReport(null);
        setShowReport(false);
        setReportError("");
        setReportSave(null);
        setHistoryDocId(null);
        historySaveRef.current = null;
        reportLoadReqRef.current++;

        const res =
          await fetch(
            "http://localhost:8000/analyze",
            {
              method:
                "POST",
              body:
                formData,
            }
          );

        if (!res.ok) {
          throw new Error(
            `서버 오류: ${res.status}`
          );
        }

        const data =
          await res.json();

        console.log(
          "백엔드 응답:",
          data
        );

        setResult(
          data
        );

        // 보고서를 기록 저장이 끝나기 전에 열어도 연결되도록 Promise 를 보관
        historySaveRef.current = saveHistory(
          data,
          thumbnailBase64
        );
        await historySaveRef.current;
      } catch (err) {
        console.error(
          "분석 결과 처리 오류:",
          err
        );

        alert(
          "분석 결과 처리 중 오류가 발생했습니다. 콘솔을 확인해주세요."
        );
      } finally {
        setUploading(
          false
        );
      }
    };

  // 기록에 저장된 보고서 불러오기 (있으면 '보고서 생성' 시 재생성 없이 바로 표시)
  const loadSavedReport = async (docId) => {
    const req = ++reportLoadReqRef.current;
    if (!docId) return;
    try {
      const saved = await fetchReport(docId);
      if (req !== reportLoadReqRef.current || !saved) return;
      setReport({
        ...saved.report,
        status: "success",
        gradcam_snapshot: saved.gradcam_snapshot || null,
      });
      setReportSave({ status: "saved", at: saved.savedAt });
    } catch (err) {
      console.error("보고서 불러오기 실패:", err);
    }
  };

  // 생성된 보고서를 현재 분석 기록에 연결해 Firestore 에 저장
  const persistReport = async (data, targetResult) => {
    if (!isDbStoreEnabled()) {
      setReportSave(null); // DB OFF: 보고서는 화면에만 표시
      return;
    }
    try {
      setReportSave({ status: "saving" });

      const docId =
        historyDocId ||
        (historySaveRef.current ? await historySaveRef.current : null);

      if (!docId) {
        setReportSave({
          status: "error",
          message: "분석 기록이 저장되지 않아 보고서를 연결할 수 없습니다.",
        });
        return;
      }

      const snapshot = await makeGradcamSnapshot(targetResult);

      const savedAt = await saveReport(docId, data, {
        gradcamSnapshot: snapshot,
        prediction: targetResult.prediction,
        confidence: targetResult.confidence,
      });

      if (savedAt === null) {
        setReportSave(null); // DB_STORE=false
        return;
      }

      setReport((prev) => (prev === data ? { ...data, gradcam_snapshot: snapshot } : prev));
      setReportSave({ status: "saved", at: savedAt });
      setReportIds((prev) => new Set(prev).add(docId));
    } catch (err) {
      console.error("보고서 저장 오류:", err);
      setReportSave({ status: "error", message: err.message || "알 수 없는 오류" });
    }
  };

    const openReport = async () => {
    if (!result) {
      alert("먼저 이미지를 분석해주세요.");
      return;
    }

    setShowReport(true);

    if (report) return; // 이미 만든(또는 저장된) 보고서는 재사용

    reportLoadReqRef.current++; // 진행 중인 불러오기 응답이 새 보고서를 덮지 않도록

    try {
      setReportLoading(true);
      setReportError("");

      const res = await fetch("http://localhost:8000/report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prediction: result.prediction,
          prediction_en: result.prediction_en,
          confidence: result.confidence,
          material: result.material,
          similarities: result.similarities,
          gradcam_summary: result.gradcam_summary || {},
          decision_path: result.decision_path || "",
          feature: result.feature,
          visual_observation_1: result.visual_observation_1,
          visual_observation_2: result.visual_observation_2,
          imaging: result.imaging || {},
          background: result.background || {},
        }),
      });

      if (!res.ok) throw new Error(`서버 오류: ${res.status}`);

      const data = await res.json();

      if (data.status !== "success") {
        throw new Error(data.message || "보고서 생성에 실패했습니다.");
      }

      setReport(data);
      persistReport(data, result); // 저장 실패해도 보고서 표시는 유지
    } catch (err) {
      console.error("보고서 생성 오류:", err);
      setReportError(err.message || "보고서 생성 중 오류가 발생했습니다.");
    } finally {
      setReportLoading(false);
    }
  };

  const openCompare =
    () => {
      if (
        selectedCompareIds.length <
        2
      ) {
        alert(
          "왼쪽 분석 기록에서 비교할 결과 2개를 선택해주세요."
        );

        return;
      }

      setCompareSummary(
        null
      );

      setShowCompareModal(
        true
      );
    };

  const handleCompareWithLLM =
    async () => {
      if (
        compareItems.length <
        2
      ) {
        alert(
          "비교할 기록을 2개 선택해주세요."
        );

        return;
      }

      try {
        setCompareLoading(
          true
        );

        setCompareSummary(
          null
        );

        const payload = {
          items:
            compareItems.map(
              (item) =>
                item.result
            ),
        };

        const res =
          await fetch(
            "http://localhost:8000/compare",
            {
              method:
                "POST",

              headers: {
                "Content-Type":
                  "application/json",
              },

              body:
                JSON.stringify(
                  payload
                ),
            }
          );

        if (!res.ok) {
          throw new Error(
            `비교 분석 오류: ${res.status}`
          );
        }

        const data =
          await res.json();

        console.log(
          "비교 API 응답:",
          data
        );

        setCompareSummary(
          data
        );
      } catch (err) {
        console.error(
          "LLM 비교 설명 오류:",
          err
        );

        alert(
          "LLM 비교 설명 생성 중 오류가 발생했습니다."
        );
      } finally {
        setCompareLoading(
          false
        );
      }
    };

  return (
    <div
      className={
        layout.page
      }
    >
      <div className="flex min-h-screen">

        {/* 사이드바 열고 닫기 */}
        <button
          onClick={() =>
            setSidebarOpen(
              !sidebarOpen
            )
          }
          className={`fixed top-5 z-50 rounded-r-xl bg-slate-900 text-white px-3 py-3 text-sm shadow-lg transition-all duration-300 ${
            sidebarOpen
              ? "left-[384px]"
              : "left-0"
          }`}
        >
          {sidebarOpen
            ? "‹"
            : "›"}
        </button>

        {/* 분석 기록 사이드바 */}
        <aside
          className={`fixed lg:sticky top-0 left-0 z-40 h-screen bg-white border-r border-slate-200 transition-all duration-300 overflow-y-auto overflow-x-hidden ${
            sidebarOpen
              ? "w-[384px] shrink-0 translate-x-0 p-5"
              : "w-0 -translate-x-full p-0 border-none"
          }`}
        >
          {activeTab === "fracture" ? (
            <>
              {/* ==========================================
                  파단면 분석 기록
              ========================================== */}

              <div className="flex items-center justify-between mb-5">
                <div>
                  <p className="text-[11px] tracking-[0.16em] uppercase text-slate-400 font-semibold">
                    Fracture History
                  </p>

                  <h2 className="text-lg font-bold mt-1 whitespace-nowrap">
                    파단면 분석 기록
                  </h2>
                </div>

                {history.length > 0 && !deleteMode && (
                  <div className="flex items-center gap-3">
                    <button
                      onClick={() => setDeleteMode(true)}
                      className="text-xs text-slate-400 hover:text-red-500 whitespace-nowrap"
                    >
                      선택 삭제
                    </button>
                    <button
                      onClick={clearHistory}
                      className="text-xs text-slate-400 hover:text-red-500 whitespace-nowrap"
                    >
                      전체 삭제
                    </button>
                  </div>
                )}
              </div>

              {!dbEnabled && (
                <p className="mb-3 rounded-lg bg-amber-50 border border-amber-200 px-3 py-2 text-[11px] text-amber-700 leading-5">
                  DB 저장이 꺼져 있습니다. 새 분석 기록은 이 화면에만 임시로 남습니다.
                </p>
              )}

              {/* 선택 삭제 */}
              {deleteMode && (
                <DeleteToolbar
                  selectedCount={deleteIds.length}
                  totalCount={history.length}
                  deleting={deleting}
                  onToggleAll={toggleDeleteSelectAll}
                  onCancel={exitDeleteMode}
                  onDelete={handleDeleteSelected}
                />
              )}

              {/* 비교 */}
              {history.length > 0 && !deleteMode && (
                <div className="mb-4 p-3 rounded-xl bg-slate-50 border border-slate-200">
                  <p className="text-xs text-slate-500 leading-5">
                    결과 2개를 선택하면 비교할 수 있습니다.
                  </p>

                  <button
                    onClick={openCompare}
                    disabled={selectedCompareIds.length < 2}
                    className="mt-2 w-full rounded-lg bg-slate-900 text-white py-2 text-xs font-semibold disabled:opacity-30 disabled:cursor-not-allowed hover:bg-slate-700 transition"
                  >
                    선택 결과 비교 ({selectedCompareIds.length}/2)
                  </button>
                </div>
              )}

              {/* 파단면 기록 목록 */}
              <div className="space-y-2">
                {history.length === 0 && (
                  <div className="rounded-xl bg-slate-50 border border-dashed p-5 text-center">
                    <p className="text-sm text-slate-400">
                      아직 파단면 분석 기록이 없습니다.
                    </p>
                  </div>
                )}

                {history.map((item) => {
                  const itemResult = item.result;

                  const rawMaterial = itemResult?.material;

                  const itemMaterial =
                    !rawMaterial ||
                    rawMaterial === "미지정" ||
                    rawMaterial === "unknown"
                      ? "미지정"
                      : (MATERIAL_LABELS[rawMaterial] || rawMaterial);

                  const checked =
                    selectedCompareIds.includes(item.id);

                  const deleteChecked =
                    deleteMode && deleteIds.includes(item.id);

                  
                  return (
                    <div
                      key={item.id}
                      onClick={() =>
                        deleteMode
                          ? toggleDeleteSelect(item.id)
                          : handleHistoryClick(item)
                      }
                      className={`relative w-[330px] max-w-full text-left p-3 rounded-xl border transition cursor-pointer ${
                        deleteChecked
                          ? "bg-red-50 border-red-400"
                          : checked && !deleteMode
                            ? "bg-blue-50 border-blue-400"
                            : "bg-white hover:border-slate-400 hover:bg-slate-50 border-slate-200"
                      }`}
                    >
                      <div className="flex gap-3">
                        {/* 이미지 */}
                        <div className="w-14 h-14 rounded-lg bg-slate-200 overflow-hidden shrink-0">
                          {item.image ? (
                            <img
                              src={item.image}
                              alt="기록 이미지"
                              className="w-full h-full object-cover"
                            />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center text-[10px] text-slate-400">
                              No Image
                            </div>
                          )}
                        </div>

                        {/* 분석 정보 */}
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center flex-wrap gap-x-2 gap-y-0.5">
                            <span className="font-bold text-sm">
                              {itemResult.display_prediction || itemResult.prediction}
                            </span>
                            <span className="text-sm text-blue-600 font-semibold">
                              {itemResult.confidence}
                            </span>
                          </div>

                          <p className="text-xs text-slate-500 truncate mt-1">
                            재질: {itemMaterial}
                          </p>

                          <p className="text-[11px] text-slate-400 mt-1">
                            {item.time}
                            {!item.docId && (
                              <span className="ml-2 rounded bg-amber-50 px-1.5 py-0.5 text-[10px] font-semibold text-amber-700">
                                임시
                              </span>
                            )}
                          </p>
                        </div>

                        {/* 오른쪽 영역: 체크박스 + 보고서 */}
                        <div className="w-11 shrink-0 flex flex-col items-end justify-between">
                          {deleteMode ? (
                            <input
                              type="checkbox"
                              checked={deleteChecked}
                              onClick={(e) => e.stopPropagation()}
                              onChange={() => toggleDeleteSelect(item.id)}
                              aria-label="삭제할 기록 선택"
                              className="w-4 h-4 accent-red-600 cursor-pointer"
                            />
                          ) : (
                            <input
                              type="checkbox"
                              checked={checked}
                              onClick={(e) => e.stopPropagation()}
                              onChange={() => toggleCompareSelect(item.id)}
                              aria-label="비교할 기록 선택"
                              className="w-4 h-4 accent-slate-900 cursor-pointer"
                            />
                          )}

                          {reportIds.has(item.docId) && (
                            <span className="rounded bg-emerald-50 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-700 whitespace-nowrap">
                              보고서
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  );

                })}
              </div>
            </>
          ) : (
            <>

              {/* ==========================================
                  Phase 분석 기록
              ========================================== */}

              <div className="flex items-center justify-between mb-5">
                <div>
                  <p className="text-[11px] tracking-[0.16em] uppercase text-emerald-600 font-semibold">
                    Phase History
                  </p>

                  <h2 className="text-lg font-bold mt-1 whitespace-nowrap">
                    Phase 분석 기록
                  </h2>
                </div>

                {phaseHistory.length > 0 && !deleteMode && (
                  <div className="flex items-center gap-3">
                    <button
                      type="button"
                      onClick={() => setDeleteMode(true)}
                      className="text-xs text-slate-400 hover:text-red-500 whitespace-nowrap"
                    >
                      선택 삭제
                    </button>
                    <button
                      type="button"
                      onClick={async () => {
                        try { await clearAllPhaseHistory(); } catch (e) { console.error(e); }
                        setPhaseHistory([]);
                      }}
                      className="text-xs text-slate-400 hover:text-red-500 whitespace-nowrap"
                    >
                      전체 삭제
                    </button>
                  </div>
                )}
              </div>

              {!dbEnabled && (
                <p className="mb-3 rounded-lg bg-amber-50 border border-amber-200 px-3 py-2 text-[11px] text-amber-700 leading-5">
                  DB 저장이 꺼져 있습니다. 새 분석 기록은 이 화면에만 임시로 남습니다.
                </p>
              )}

              {deleteMode && (
                <DeleteToolbar
                  selectedCount={deleteIds.length}
                  totalCount={phaseHistory.length}
                  deleting={deleting}
                  onToggleAll={toggleDeleteSelectAll}
                  onCancel={exitDeleteMode}
                  onDelete={handleDeleteSelected}
                />
              )}

              <div className="space-y-2">

                {/* 기록이 없을 때 */}
                {phaseHistory.length === 0 && (
                  <div className="rounded-xl bg-emerald-50/50 border border-dashed border-emerald-200 p-5 text-center">

                    <div className="w-10 h-10 mx-auto mb-3 rounded-full bg-emerald-100 flex items-center justify-center text-emerald-700 font-bold">
                      P
                    </div>

                    <p className="text-sm font-semibold text-slate-600">
                      아직 Phase 분석 기록이 없습니다.
                    </p>

                    <p className="text-xs text-slate-400 mt-2 leading-5">
                      Phase 분석을 실행하면
                      여기에 기록됩니다.
                    </p>

                  </div>
                )}

                {/* Phase 분석 기록 */}
                {phaseHistory.map((item) => {

                const deleteChecked =
                  deleteMode && deleteIds.includes(item.id);

                return (
                    <div
                      key={item.id}
                      onClick={() =>
                        deleteMode
                          ? toggleDeleteSelect(item.id)
                          : handlePhaseHistoryClick(item)
                      }
                      className={`relative w-full p-3 rounded-xl border transition cursor-pointer ${
                        deleteChecked
                          ? "bg-red-50 border-red-400"
                          : "border-slate-200 bg-white hover:border-emerald-400 hover:bg-emerald-50/30"
                      }`}
                    >

                      {deleteMode && (
                        <input
                          type="checkbox"
                          checked={deleteChecked}
                          onClick={(e) => e.stopPropagation()}
                          onChange={() => toggleDeleteSelect(item.id)}
                          aria-label="삭제할 기록 선택"
                          className="absolute top-3 right-3 w-4 h-4 accent-red-600 cursor-pointer"
                        />
                      )}

                      <div className={`flex gap-3 ${deleteMode ? "pr-6" : ""}`}>

                        {/* 원본 이미지 */}
                        <div className="w-14 h-14 rounded-lg bg-slate-200 overflow-hidden shrink-0">

                          {item.image ? (
                            <img
                              src={item.image}
                              alt="Phase 기록 이미지"
                              className="w-full h-full object-cover"
                            />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center text-[10px] text-slate-400">
                              No Image
                            </div>
                          )}

                        </div>

                        {/* 분석 정보 */}
                        <div className="min-w-0 flex-1">

                          <div className="flex items-start justify-between gap-2">

                            <p className="text-sm font-bold text-slate-700 truncate">
                              {item.title || "Phase 분석"}
                            </p>

                            {!deleteMode && (
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();

                                handlePhaseHistoryRename(
                                  item.id,
                                  item.title || "Phase 분석"
                                );
                              }}
                              className="shrink-0 text-xs text-slate-400 hover:text-emerald-600 transition"
                              title="제목 변경"
                            >
                              수정
                            </button>
                            )}

                          </div>

                          <p className="text-[11px] text-slate-400 mt-1">
                            {item.time}
                            {!item.docId && (
                              <span className="ml-2 rounded bg-amber-50 px-1.5 py-0.5 text-[10px] font-semibold text-amber-700">
                                임시
                              </span>
                            )}
                          </p>

                        </div>

                      </div>

                    </div>
                  );
                })}

              </div>
            </>
          )}
        </aside>

        {/* 메인 */}
        <main className="flex-1 min-w-0">

          {/* 상단 바 */}
        <header className="bg-[#172536] text-white border-b border-slate-700">
          <div
            className={`${layout.container} h-[68px] grid grid-cols-[1fr_auto_1fr] items-center`}
          >
            {/* 로고 */}
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-full border border-white/30 flex items-center justify-center text-lg">
                ◈
              </div>

              <div>
                <h1 className="font-bold text-lg leading-none">
                  Fracture Analysis System
                </h1>

                <p className="text-[10px] text-slate-300 mt-1 tracking-[0.14em] uppercase">
                  AI Fractography Analysis
                </p>
              </div>
            </div>

            {/* 분석 종류 탭 */}
            <div className="flex items-center bg-white/10 p-1 rounded-xl border border-white/10">
              <button
                onClick={() => setActiveTab("fracture")}
                className={`min-w-[150px] px-6 py-2 rounded-lg text-sm font-bold transition ${
                  activeTab === "fracture"
                    ? "bg-blue-600 text-white shadow-md"
                    : "text-slate-300 hover:text-white hover:bg-white/5"
                }`}
              >
                파단면 분석
              </button>

              <button
                onClick={() => setActiveTab("phase")}
                className={`min-w-[150px] px-6 py-2 rounded-lg text-sm font-bold transition ${
                  activeTab === "phase"
                    ? "bg-blue-600 text-white shadow-md"
                    : "text-slate-300 hover:text-white hover:bg-white/5"
                }`}
              >
                Phase 분석
              </button>
            </div>

            {/* DB 저장 On/Off */}
            <div className="flex justify-end">
              <DbStoreToggle enabled={dbEnabled} onChange={handleDbToggle} />
            </div>
          </div>
        </header>

          <div
            className={`${layout.container} py-8`}
          >
            {activeTab === "fracture" ? (
              <>

            {/* 제목 */}
            <div className="mb-6">
              <p className="text-xs uppercase tracking-[0.18em] text-blue-600 font-bold">
                Failure Analysis
              </p>

              <h2 className="text-2xl font-bold mt-1">
                파손단면 이미지 분석
              </h2>

              <p className="text-sm text-slate-500 mt-2">
                입력 이미지와 모델의
                주요 판단 결과를 한
                화면에서 확인합니다.
              </p>
            </div>

            {/* 메인 분석 카드 */}
            <section className="bg-white rounded-[22px] border border-slate-200 shadow-sm overflow-hidden">

              <div className="grid xl:grid-cols-[1fr_1fr_0.82fr] min-h-[665px]">

                {/* 입력 이미지 */}
                <div className="p-5 border-b xl:border-b-0 xl:border-r border-slate-200 flex flex-col">

                  <div className="flex items-start justify-between">
                    <div>
                      <h3 className="font-bold text-base">
                        입력 이미지
                      </h3>
                      <p className="text-xs text-slate-400 mt-1">
                        분석 대상 원본 이미지
                      </p>
                    </div>

                    {previewUrl && (
                      <button
                        onClick={() => fileRef.current?.click()}
                        className="text-xs text-blue-600 font-semibold hover:text-blue-800"
                      >
                        이미지 변경
                      </button>
                    )}
                  </div>

                  <div className="flex-1 flex flex-col justify-start pt-10 pb-5">
                    <label
                      htmlFor="file-input"
                      className="h-[350px] rounded-xl border border-slate-200 bg-slate-50 overflow-hidden flex items-center justify-center cursor-pointer"
                    >
                      {previewUrl ? (
                        <img
                          src={previewUrl}
                          alt="입력 이미지"
                          className="w-full h-full object-contain"
                        />
                      ) : (
                        <div className="text-center px-5">
                          <div className="w-12 h-12 rounded-full bg-slate-200 mx-auto flex items-center justify-center text-xl mb-3">
                            +
                          </div>
                          <p className="font-semibold text-slate-700">
                            파손단면 이미지를 업로드하세요
                          </p>
                          <p className="text-xs text-slate-400 mt-2">
                            클릭하여 이미지 선택
                          </p>
                        </div>
                      )}
                    </label>

                    <input
                      id="file-input"
                      type="file"
                      accept="image/*"
                      ref={fileRef}
                      onChange={handleFileChange}
                      className="hidden"
                    />

                    <div className="mt-4 flex gap-3">
                      <button
                        onClick={handleUpload}
                        disabled={uploading}
                        className="flex-1 h-11 rounded-lg bg-[#172536] text-white text-sm font-semibold hover:bg-slate-700 disabled:opacity-50 transition"
                      >
                        {uploading ? "분석 중..." : "분석 시작"}
                      </button>
                    </div>

                    <details className="mt-3 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                      <summary className="cursor-pointer text-xs text-slate-600">
                        사전 배경 정보 (선택) · {backgroundSummary(background, material)}
                      </summary>

                      <div className="mt-3 flex flex-col gap-3 text-xs">
                        <label className="flex flex-col gap-1">
                          <span className="text-slate-500">재질</span>
                          <select
                            value={material}
                            onChange={(e) => setMaterial(e.target.value)}
                            className="h-9 px-2 rounded border border-slate-300 bg-white"
                          >
                            <option value="unknown">미지정</option>
                            {Object.entries(MATERIAL_LABELS)
                              .filter(([value]) => value !== "unknown")
                              .map(([value, label]) => (
                                <option key={value} value={value}>
                                  {label}
                                </option>
                              ))}
                          </select>
                        </label>

                        <div>
                          <p className="text-slate-500 mb-1">사용 환경 (복수 선택)</p>
                          <div className="grid grid-cols-2 gap-1">
                            {Object.entries(BG_ENVIRONMENT_LABELS).map(([v, l]) => (
                              <label key={v} className="flex items-start gap-1.5">
                                <input
                                  type="checkbox"
                                  className="mt-0.5"
                                  checked={background.environment.includes(v)}
                                  onChange={() => toggleBackground("environment", v)}
                                />
                                <span>{l}</span>
                              </label>
                            ))}
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-2">
                          <label className="flex flex-col gap-1">
                            <span className="text-slate-500">하중 형태</span>
                            <select
                              value={background.load}
                              onChange={updateBackground("load")}
                              className="h-9 px-2 rounded border border-slate-300 bg-white"
                            >
                              <option value="">미지정</option>
                              {Object.entries(BG_LOAD_LABELS).map(([v, l]) => (
                                <option key={v} value={v}>{l}</option>
                              ))}
                            </select>
                          </label>

                          <label className="flex flex-col gap-1">
                            <span className="text-slate-500">사용 온도</span>
                            <select
                              value={background.temperature}
                              onChange={updateBackground("temperature")}
                              className="h-9 px-2 rounded border border-slate-300 bg-white"
                            >
                              <option value="">미지정</option>
                              {Object.entries(BG_TEMPERATURE_LABELS).map(([v, l]) => (
                                <option key={v} value={v}>{l}</option>
                              ))}
                            </select>
                          </label>

                          <label className="flex flex-col gap-1 col-span-2">
                            <span className="text-slate-500">파손 양상</span>
                            <select
                              value={background.failure_manner}
                              onChange={updateBackground("failure_manner")}
                              className="h-9 px-2 rounded border border-slate-300 bg-white"
                            >
                              <option value="">미지정</option>
                              {Object.entries(BG_MANNER_LABELS).map(([v, l]) => (
                                <option key={v} value={v}>{l}</option>
                              ))}
                            </select>
                          </label>
                        </div>

                        <div>
                          <p className="text-slate-500 mb-1">제조·처리 이력 (복수 선택)</p>
                          <div className="grid grid-cols-2 gap-1">
                            {Object.entries(BG_PROCESS_LABELS).map(([v, l]) => (
                              <label key={v} className="flex items-center gap-1.5">
                                <input
                                  type="checkbox"
                                  checked={background.process.includes(v)}
                                  onChange={() => toggleBackground("process", v)}
                                />
                                <span>{l}</span>
                              </label>
                            ))}
                          </div>
                        </div>

                        <label className="flex flex-col gap-1">
                          <span className="text-slate-500">메모 (사용 기간, 시험 조건 등 · 200자)</span>
                          <textarea
                            rows={2}
                            maxLength={200}
                            placeholder="예) 수소 충전 144시간 후 인장 시험"
                            value={background.note}
                            onChange={updateBackground("note")}
                            className="px-2 py-1.5 rounded border border-slate-300 bg-white resize-none"
                          />
                        </label>
                      </div>

                      <div className="mt-2 flex items-center justify-between">
                        <p className="text-[11px] text-slate-400">
                          원인 추정에만 쓰이며, 분류 결과는 바뀌지 않습니다.
                        </p>
                        <button
                          type="button"
                          onClick={() => {
                            setBackground(EMPTY_BACKGROUND);
                            setMaterial("");
                          }}
                          className="text-[11px] text-slate-500 underline"
                        >
                          모두 미지정으로
                        </button>
                      </div>
                    </details>

                    <details className="mt-3 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                      <summary className="cursor-pointer text-xs text-slate-600">
                        SEM 촬영 정보 (선택) · {imagingSummary(imaging)}
                      </summary>

                      <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                        <label className="flex flex-col gap-1">
                          <span className="text-slate-500">배율 (x)</span>
                          <input
                            type="number"
                            min="0"
                            placeholder="미지정"
                            value={imaging.magnification}
                            onChange={updateImaging("magnification")}
                            className="h-9 px-2 rounded border border-slate-300 bg-white"
                          />
                        </label>

                        <label className="flex flex-col gap-1">
                          <span className="text-slate-500">스케일 바 (µm)</span>
                          <input
                            type="number"
                            min="0"
                            step="any"
                            placeholder="미지정"
                            value={imaging.scale_bar_um}
                            onChange={updateImaging("scale_bar_um")}
                            className="h-9 px-2 rounded border border-slate-300 bg-white"
                          />
                        </label>

                        <label className="flex flex-col gap-1">
                          <span className="text-slate-500">촬영 위치</span>
                          <select
                            value={imaging.location}
                            onChange={updateImaging("location")}
                            className="h-9 px-2 rounded border border-slate-300 bg-white"
                          >
                            <option value="">미지정</option>
                            {Object.entries(SEM_LOCATION_LABELS).map(([v, l]) => (
                              <option key={v} value={v}>{l}</option>
                            ))}
                          </select>
                        </label>

                        <label className="flex flex-col gap-1">
                          <span className="text-slate-500">검출기</span>
                          <select
                            value={imaging.detector}
                            onChange={updateImaging("detector")}
                            className="h-9 px-2 rounded border border-slate-300 bg-white"
                          >
                            <option value="">미지정</option>
                            {Object.entries(SEM_DETECTOR_LABELS).map(([v, l]) => (
                              <option key={v} value={v}>{l}</option>
                            ))}
                          </select>
                        </label>

                        <label className="flex flex-col gap-1">
                          <span className="text-slate-500">세척</span>
                          <select
                            value={imaging.cleaned}
                            onChange={updateImaging("cleaned")}
                            className="h-9 px-2 rounded border border-slate-300 bg-white"
                          >
                            <option value="">미지정</option>
                            {Object.entries(YES_NO_LABELS).map(([v, l]) => (
                              <option key={v} value={v}>{l}</option>
                            ))}
                          </select>
                        </label>

                        <label className="flex flex-col gap-1">
                          <span className="text-slate-500">도전성 코팅</span>
                          <select
                            value={imaging.coated}
                            onChange={updateImaging("coated")}
                            className="h-9 px-2 rounded border border-slate-300 bg-white"
                          >
                            <option value="">미지정</option>
                            {Object.entries(YES_NO_LABELS).map(([v, l]) => (
                              <option key={v} value={v}>{l}</option>
                            ))}
                          </select>
                        </label>
                      </div>

                      <div className="mt-2 flex items-center justify-between">
                        <p className="text-[11px] text-slate-400">
                          입력한 항목만 보고서에 참고됩니다.
                        </p>
                        <button
                          type="button"
                          onClick={() => setImaging(EMPTY_IMAGING)}
                          className="text-[11px] text-slate-500 underline"
                        >
                          모두 미지정으로
                        </button>
                      </div>
                    </details>
                  </div>
                </div>

                {/* Grad-CAM++ */}
                <div className="p-5 border-b xl:border-b-0 xl:border-r border-slate-200 flex flex-col">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-bold text-base">
                        Grad-CAM++ 결과
                      </h3>
                      <p className="text-xs text-slate-400 mt-1">
                        모델이 판단에 활용한 영역
                      </p>
                    </div>

                    {result &&
                      (result.gradcam_masks ||
                        result.gradcam_layers ||
                        (result.gradcam_contours &&
                          Object.keys(result.gradcam_contours).length > 0)) && (
                        <button
                          onClick={() => setShowGradcamModal(true)}
                          className="text-xs px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 transition"
                        >
                          확대 보기
                        </button>
                      )}
                  </div>

                  <div className="flex-1 flex flex-col justify-start pt-10 pb-5">
                    {result &&
                    (result.gradcam_masks ||
                      result.gradcam_layers ||
                      (result.gradcam_contours &&
                        Object.keys(result.gradcam_contours).length > 0)) ? (
                      // <div className="mt-7">
                        <GradcamView
                          result={result}
                          chipSize="text-[11px]"
                          canvasClass="h-[350px]"
                        />
                      // </div>
                    ) : (
                      <div className="h-[350px] rounded-xl border border-dashed border-slate-300 bg-slate-50 flex items-center justify-center">
                        <div className="text-center">
                          <p className="text-sm font-semibold text-slate-500">
                            분석 전
                          </p>
                          <p className="text-xs text-slate-400 mt-2">
                            분석을 실행하면 시각화 결과가 표시됩니다.
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* 분석 결과 */}
                <div className="p-5 flex flex-col">
                  <div className="flex items-center justify-between gap-2">
                    <h3 className="font-bold text-base">
                      분석 결과
                    </h3>
                    <button
                      onClick={openReport}
                      disabled={!result || reportLoading}
                      className="px-3 py-2 rounded-lg bg-blue-600 text-white text-xs font-bold hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      {reportLoading ? "생성 중..." : "AI 보고서 생성"}
                    </button>
                  </div>

                  {result ? (
                    <div className="flex-1 flex flex-col pt-4">
                      <div className="rounded-xl bg-gradient-to-br from-blue-50 to-slate-50 border border-blue-100 py-5 px-5 text-center">
                        <p className="text-xs text-slate-500 uppercase tracking-[0.16em]">
                          Predicted fracture
                        </p>
                        <p className="text-3xl font-black text-blue-700 mt-2">
                          {result.display_prediction || result.prediction}
                        </p>
                        <p className="text-4xl font-black text-slate-900 mt-1">
                          {result.confidence}
                        </p>
                        <span
                          className={`inline-block mt-3 px-3 py-1 rounded-full border text-xs font-bold ${confidenceStyle}`}
                        >
                          예측 점수 {confidenceLabel}
                        </span>
                      </div>

                      {result.similarities && (
                        <div className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
                          <p className="text-xs font-bold text-slate-600 mb-3">
                            전체 예측 확률
                          </p>

                          <div className="space-y-3">
                            {Object.entries(result.similarities)
                              .sort((a, b) => (parseFloat(b[1]) || 0) - (parseFloat(a[1]) || 0))
                              .map(([name, value]) => {
                                const percent = Math.max(
                                  0,
                                  Math.min(100, parseFloat(value) || 0)
                                );
                                const isPredicted =
                                  name === (result.display_prediction || result.prediction) ||
                                  EN_NAMES[name] === result.prediction_en;

                                return (
                                  <div key={name}>
                                    <div className="flex justify-between text-[11px] mb-1">
                                      <span className={isPredicted ? "font-bold" : "text-slate-500"}>
                                        {name}
                                      </span>
                                      <span className={isPredicted ? "font-bold text-blue-700" : "text-slate-500"}>
                                        {typeof value === "number" ? `${value.toFixed(1)}%` : value}
                                      </span>
                                    </div>
                                    <div className="h-1.5 rounded-full bg-slate-100 overflow-hidden">
                                      <div
                                        className={`h-full rounded-full ${isPredicted ? "bg-blue-600" : "bg-slate-300"}`}
                                        style={{ width: `${percent}%` }}
                                      />
                                    </div>
                                  </div>
                                );
                              })}
                          </div>
                        </div>
                      )}

                      <div className="mt-4 grid grid-cols-2 gap-3">
                        <div className="rounded-lg border border-slate-200 p-3">
                          <p className="text-[11px] text-slate-400">재질</p>
                          <p className="text-sm font-bold mt-1">{materialText}</p>
                        </div>
                        <div className="rounded-lg border border-slate-200 p-3">
                          <p className="text-[11px] text-slate-400">상태</p>
                          <p className="text-sm font-bold mt-1">{confidenceLabel}</p>
                        </div>
                      </div>

                      <div className={`mt-3 rounded-lg border p-3 ${confidenceStyle}`}>
                        <p className="text-xs leading-5">
                          {result.confidence_message}
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="flex-1 flex items-center justify-center">
                      <div className="w-full h-[350px] rounded-xl bg-slate-50 border border-dashed border-slate-300 flex items-center justify-center">
                        <p className="text-sm text-slate-500">
                          이미지를 선택하고 분석을 시작하세요.
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* 하단 기능 탭 */}
              <div className="border-t border-slate-200 p-5">
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { key: "rag", label: "AI 설명 참고 자료" },
                    { key: "similar", label: "유사 이미지" },
                  ].map((tab) => (
                    <button
                      key={tab.key}
                      onClick={() => setDetailTab(tab.key)}
                      className={`rounded-xl px-4 py-3 text-sm font-bold border transition ${
                        detailTab === tab.key
                          ? "bg-[#172536] text-white border-[#172536]"
                          : "bg-white text-slate-600 border-slate-200 hover:bg-slate-50"
                      }`}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>

                {result && (
                  <div className="mt-5 rounded-xl border border-slate-200 p-5">
                    {detailTab === "rag" && (
                      <ReferenceCard
                        references={result.references || []}
                        knowledgeRefs={result.knowledge_refs || []}
                        reportLoading={false}
                      />
                    )}
                    
                    {detailTab === "similar" && (
                      <div>
                        <h3 className="font-bold mb-4">
                          유사 이미지 Top-3
                        </h3>

                        {result.similar_images?.length > 0 ? (
                          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                            {result.similar_images.slice(0, 3).map((item, index) => (
                              <div
                                key={index}
                                className="rounded-xl border border-slate-200 overflow-hidden bg-white"
                              >
                                <div className="h-48 bg-slate-50">
                                  <img
                                    src={
                                      item.image_url?.startsWith("http")
                                        ? item.image_url
                                        : `http://localhost:8000${item.image_url}`
                                    }
                                    alt={`유사 이미지 ${index + 1}`}
                                    className="w-full h-full object-contain"
                                  />
                                </div>

                                <div className="p-4">
                                  <p className="font-bold text-sm">
                                    Top {index + 1}
                                  </p>

                                  <p className="text-xs text-slate-500 mt-1">
                                    유형: {item.class_name || item.label || "-"}
                                  </p>

                                  <p className="text-sm font-semibold text-blue-600 mt-2">
                                    유사도: {
                                      typeof item.similarity === "number"
                                        ? `${(item.similarity * 100).toFixed(1)}%`
                                        : item.similarity || "-"
                                    }
                                  </p>
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <p className="text-sm text-slate-400 py-6 text-center">
                            유사 이미지 결과가 없습니다.
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </section>
              </>
            ) : (
              <>
                {/* Phase 분석 */}
                <div className="mb-6 flex flex-col lg:flex-row lg:items-start lg:justify-between gap-4">
                  <div>
                    <p className="text-xs uppercase tracking-[0.18em] text-emerald-600 font-bold">
                      Phase Analysis
                    </p>
                    <h2 className="text-2xl font-bold mt-1">
                      Phase 이미지 분석
                    </h2>
                    <p className="text-sm text-slate-500 mt-2">
                      현미경 조직 이미지에서 영역별 Phase를 분류하고 분포를 확인합니다.
                    </p>
                  </div>

                  <div className="rounded-xl border border-emerald-100 bg-emerald-50 px-4 py-3 shrink-0">
                    <p className="text-xs font-semibold text-emerald-800">
                      Phase 분석은 별도의 현미경 조직 이미지를 사용합니다.
                    </p>
                    <p className="text-xs text-slate-500 mt-1">
                      파단면 이미지와 Phase 분석용 조직 이미지는 서로 다른 입력입니다.
                    </p>
                  </div>
                </div>
           
                {/* Phase 분석 메인 카드 */}
                <section className="bg-white rounded-[22px] border border-slate-200 shadow-sm overflow-hidden">
                  <div className="grid xl:grid-cols-3">

                    {/* 1. 입력 이미지 */}
                    <div className="p-5 border-b xl:border-b-0 xl:border-r border-slate-200 flex flex-col">
                      <div className="flex items-center justify-between mb-5">
                        <div>
                          <h3 className="font-bold text-base">입력 이미지</h3>
                          <p className="text-xs text-slate-400 mt-1">
                            Phase 분석용 현미경 조직 이미지
                          </p>
                        </div>

                        {phasePreviewUrl && (
                          <button
                            onClick={() => phaseFileRef.current?.click()}
                            className="text-xs text-emerald-600 font-semibold"
                          >
                            이미지 변경
                          </button>
                        )}
                      </div>

                      <label
                        htmlFor="phase-file-input"
                        className="h-[350px] rounded-xl border border-slate-200 bg-slate-50 flex items-center justify-center overflow-hidden cursor-pointer"
                      >
                        {phasePreviewUrl ? (
                          <img
                            src={phasePreviewUrl}
                            alt="Phase 입력"
                            className="w-full h-full object-contain"
                          />
                        ) : (
                          <p className="text-sm text-slate-400 text-center px-4">
                            현미경 조직 이미지를 업로드하세요
                          </p>
                        )}
                      </label>

                      <input
                        id="phase-file-input"
                        type="file"
                        accept="image/*"
                        ref={phaseFileRef}
                        onChange={handlePhaseFileChange}
                        className="hidden"
                      />

                      <button
                        onClick={handlePhaseAnalyze}
                        disabled={!phaseFile || phaseLoading}
                        className="mt-4 h-11 bg-[#172536] text-white rounded-lg font-semibold disabled:opacity-40"
                      >
                        {phaseLoading ? "Phase 분석 중..." : "Phase 분석 시작"}
                      </button>
                    </div>

                    {/* 2. Segmentation 결과 */}
                    <div className="p-5 border-b xl:border-b-0 xl:border-r border-slate-200">
                      <div className="mb-5">
                        <h3 className="font-bold text-base">
                          Phase Segmentation 결과
                        </h3>
                        <p className="text-xs text-slate-400 mt-1">
                          AI가 분류한 영역별 Phase 시각화
                        </p>
                      </div>

                      {/* Segmentation 결과 이미지 */}
                      <div className="h-[350px] rounded-xl border border-slate-200 bg-slate-50 flex items-center justify-center overflow-hidden">
                        {phaseResult?.overlay_image ? (
                          <img
                            src={phaseResult.overlay_image}
                            alt="Phase Segmentation"
                            className="w-full h-full object-contain"
                          />
                        ) : (
                          <p className="text-sm text-slate-400">
                            {phaseLoading ? "분석 중..." : "분석 결과 없음"}
                          </p>
                        )}
                      </div>

                      {/* Phase 색상 범례 */}
                      <div className="flex flex-wrap gap-x-4 gap-y-2 mt-4">
                        {[
                          ["Al", "Al"],
                          ["Primary Si", "Primary Si"],
                          ["Al₃Ni", "Al3Ni"],
                          ["Eutectic Si", "Eutectic Si"],
                        ].map(([label, key]) => (
                          <div key={key} className="flex items-center gap-2">
                            <span
                              className="w-3 h-3 rounded-sm border border-slate-200"
                              style={{ backgroundColor: PHASE_COLORS[key] }}
                            />
                            <span className="text-xs text-slate-600">
                              {label}
                            </span>
                          </div>
                        ))}
                      </div>

                      {/* Phase 분석 오류 */}
                      {phaseError && (
                        <p className="text-xs text-red-600 mt-3">
                          {phaseError}
                        </p>
                      )}
                    </div>

                    {/* 3. Phase 분석 결과 */}
                    <div className="p-5">
                      <div className="mb-5">
                        <h3 className="font-bold text-base">
                          Phase 분석 결과
                        </h3>
                        <p className="text-xs text-slate-400 mt-1">
                          Phase별 면적 구성 비율
                        </p>
                      </div>

                      {/* 주요 Phase 자동 요약 */}
                      {phaseResult?.phase_distribution && (() => {
                        const entries = Object.entries(
                          phaseResult.phase_distribution
                        ).filter(([, value]) => typeof value === "number");

                        if (entries.length === 0) return null;

                        const [name, percent] = entries.reduce(
                          (max, current) =>
                            current[1] > max[1] ? current : max
                        );

                        return (
                          <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-4">
                            <p className="text-xs font-semibold text-emerald-700">
                              가장 높은 면적 비율
                            </p>

                            <div className="flex items-center justify-between mt-2">
                              <p className="text-lg font-bold text-slate-800">
                                {name === "Al3Ni" ? "Al₃Ni" : name}
                              </p>

                              <p className="text-xl font-bold text-emerald-700">
                                {percent.toFixed(1)}%
                              </p>
                            </div>
                          </div>
                        );
                      })()}

                      <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 mt-5">
                        <h4 className="text-xs font-bold text-slate-600 mb-4">
                          Phase 구성 비율
                        </h4>

                        <div className="space-y-5">
                          {[
                            ["Primary Si", "Primary Si"],
                            ["Eutectic Si", "Eutectic Si"],
                            ["Al₃Ni", "Al3Ni"],
                            ["Al", "Al"],
                          ].map(([label, key]) => {
                            const value =
                              phaseResult?.phase_distribution?.[key];

                            const percent =
                              typeof value === "number" ? value : 0;

                            return (
                              <div key={key}>
                                <div className="flex justify-between text-xs mb-2">
                                  <span className="text-slate-500">{label}</span>
                                  <span className="font-semibold text-slate-700">
                                    {typeof value === "number"
                                      ? `${value.toFixed(1)}%`
                                      : "-"}
                                  </span>
                                </div>

                                <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                                  <div
                                    className="h-full rounded-full"
                                    style={{
                                      width: `${percent}%`,
                                      backgroundColor: PHASE_COLORS[key],
                                    }}
                                  />
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    </div>

                  </div>
                </section>

                {/* Phase 분석 하단 안내 */}
                <div className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
                  <p className="text-xs text-slate-500 leading-6">
                    <span className="text-emerald-600 mr-2">●</span>
                    Phase 분석은 파단면 분석과 별도의 입력 이미지와 모델을 사용합니다.
                    분석 결과에는 영역별 Phase 분류 이미지와 각 Phase의 면적 비율이
                    표시됩니다. 면적 비율은 이미지에서 각 Phase가 차지하는 픽셀
                    비율이며, 모델의 예측 신뢰도와는 다른 값입니다.
                  </p>
                </div>
              </>
            )}
          </div>
        </main>
      </div>

      {showReport && result && (
        <ReportModal
          result={result}
          previewUrl={previewUrl}
          report={report}
          loading={reportLoading}
          error={reportError}
          saveState={reportSave}
          onClose={() => setShowReport(false)}
        />
      )}

      {showSimilarModal && result && (
        <ModalShell
          title="유사 이미지"
          subtitle="입력 이미지와 특징이 유사한 사례 Top-3"
          onClose={() => setShowSimilarModal(false)}
        >
          {result.similar_images?.length > 0 ? (
            <div className="grid md:grid-cols-3 gap-5">
              {result.similar_images.slice(0, 3).map((item, index) => (
                <div key={index} className="rounded-xl border overflow-hidden">
                  <img
                    src={`http://localhost:8000${item.image_url}`}
                    alt={`유사 사례 ${index + 1}`}
                    className="w-full h-52 object-cover"
                  />
                  <div className="p-4">
                    <p className="font-bold">
                      TOP {index + 1} · {item.class_name || item.label}
                    </p>
                    <p className="text-sm text-slate-500 mt-2">
                      {typeof item.similarity === "number"
                        ? `${(item.similarity * 100).toFixed(1)}%`
                        : item.similarity}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-slate-400">
              유사 이미지 결과가 없습니다.
            </p>
          )}
        </ModalShell>
      )}

      {showGradcamModal && result && (
        <ModalShell
          title="Grad-CAM++ 확대 보기"
          onClose={() => setShowGradcamModal(false)}
        >
          <GradcamView
            result={result}
            chipSize="text-sm"
            canvasClass="max-h-[70vh] min-h-[500px]"
          />
        </ModalShell>
      )}

      {showCompareModal && (
        <ModalShell
          title="분석 결과 비교"
          subtitle="선택한 두 분석 결과의 특징과 차이를 비교합니다."
          onClose={() => setShowCompareModal(false)}
          maxWidth="max-w-6xl"
        >
          {compareItems.length < 2 ? (
            <p className="text-center text-slate-500 p-10">
              비교할 기록 2개를 선택해주세요.
            </p>
          ) : (
            <>
              <div className="grid md:grid-cols-2 gap-5">
                {compareItems.map((item, index) => (
                  <div key={item.id} className="rounded-xl border p-5">
                    <h3 className="font-bold mb-3">이미지 {index + 1}</h3>
                    {item.image && (
                      <img
                        src={item.image}
                        alt={`비교 이미지 ${index + 1}`}
                        className="w-full h-56 object-contain bg-slate-50 rounded-lg"
                      />
                    )}
                    <p className="font-bold mt-4">
                      {item.result.display_prediction || item.result.prediction}
                    </p>
                    <p className="text-blue-600 font-bold">
                      {item.result.confidence}
                    </p>
                    <p className="text-sm mt-3">{item.result.feature}</p>
                    <p className="text-sm mt-3">{item.result.expected_cause}</p>
                  </div>
                ))}
              </div>

              <div className="text-center mt-6">
                <button
                  onClick={handleCompareWithLLM}
                  disabled={compareLoading}
                  className="bg-[#172536] text-white px-7 py-3 rounded-xl font-bold disabled:opacity-50"
                >
                  {compareLoading ? "비교 설명 생성 중..." : "AI 비교 설명 생성"}
                </button>
              </div>

              {compareSummary && (
                <div className="mt-7 space-y-4">
                  {[
                    ["핵심 요약", compareSummary.summary || compareSummary.compare_summary],
                    ["공통점", compareSummary.common_point],
                    ["시각적 특징 차이", compareSummary.visual_difference],
                    ["파손 메커니즘 차이", compareSummary.mechanism_difference],
                    ["예상 원인 차이", compareSummary.cause_difference],
                    ["예측 확률 비교", compareSummary.confidence_difference],
                  ].map(([title, value]) => (
                    <div key={title} className="rounded-xl border p-5">
                      <h3 className="font-bold">{title}</h3>
                      <p className="text-sm leading-7 mt-2 text-slate-600">
                        {value || "정보가 없습니다."}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </ModalShell>
      )}
    </div>
  );
}
