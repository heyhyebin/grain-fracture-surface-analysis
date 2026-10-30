"""
패치 크기 비교 평가 스크립트 (fracture_surface/backend 폴더에 두고 실행)

정답 폴더 구조 (클래스 이름은 모델과 동일하게):
    test_set/
      Cleavage/ *.png|jpg
      Ductile/
      Fatigue/
      Intergranular/
    (없는 클래스 폴더는 생략 가능)

실행:
    python eval_patch_size.py --data test_set
    python eval_patch_size.py --data test_set --configs 384:192 224:112 whole

출력: 설정별 정확도 / 클래스별 재현율 / 평균 1위 확률 / 이미지당 추론 시간,
      그리고 이미지별 결과 CSV (patch_eval_results.csv)
"""
import argparse, csv, os, time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from model import load_model
from gradcam import predict_sliding

CLASSES = ["Cleavage", "Ductile", "Fatigue", "Intergranular"]
EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def parse_config(text, base):
    cfg = dict(base)
    if text == "whole":                      # 이미지 전체를 한 장으로 (학습 val 방식과 가장 비슷)
        cfg["patch_size"], cfg["stride"] = None, None
    else:
        p, s = text.split(":")
        cfg["patch_size"], cfg["stride"] = int(p), int(s)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=os.path.join("model", "fractography_best.pth"))
    ap.add_argument("--configs", nargs="+", default=["384:192", "224:112", "whole"])
    ap.add_argument("--min-side", type=int, default=0,
                    help="이 값보다 작은 이미지를 확대 (패치 224 브랜치의 확대 동작 재현용, 0이면 끔)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, base_cfg = load_model(args.model, device, num_classes=len(CLASSES))

    samples = []
    for label in CLASSES:
        d = os.path.join(args.data, label)
        if os.path.isdir(d):
            samples += [(os.path.join(d, f), label) for f in sorted(os.listdir(d)) if f.lower().endswith(EXTS)]
    if not samples:
        raise SystemExit(f"{args.data} 아래에 클래스 폴더/이미지가 없습니다.")
    print(f"이미지 {len(samples)}장, 설정 {args.configs}\n")

    rows = []
    for text in args.configs:
        cfg = parse_config(text, base_cfg)
        correct, n_by, ok_by, top_probs, times = 0, defaultdict(int), defaultdict(int), [], []
        for path, label in samples:
            img = Image.open(path).convert("RGB")
            if args.min_side and min(img.size) < args.min_side:
                s = args.min_side / min(img.size)
                img = img.resize((round(img.width * s), round(img.height * s)), Image.Resampling.LANCZOS)
            t0 = time.perf_counter()
            probs, patch_probs, boxes = predict_sliding(model, img, cfg, device)
            times.append(time.perf_counter() - t0)
            pred = CLASSES[int(np.argmax(probs))]
            hit = pred == label
            correct += hit; n_by[label] += 1; ok_by[label] += hit
            top_probs.append(float(probs.max()))
            # 패치 간 일치도: 패치별 1위 클래스 중 최빈값 비율 (낮으면 패치마다 판단이 갈림)
            agree = np.bincount(patch_probs.argmax(1), minlength=len(CLASSES)).max() / len(patch_probs)
            rows.append({"config": text, "file": os.path.basename(path), "label": label, "pred": pred,
                         "correct": int(hit), "top_prob": round(float(probs.max()) * 100, 1),
                         "n_patches": len(boxes), "patch_agreement": round(float(agree), 2),
                         **{f"p_{c}": round(float(p) * 100, 1) for c, p in zip(CLASSES, probs)}})

        recalls = {c: ok_by[c] / n_by[c] for c in CLASSES if n_by[c]}
        print(f"[{text:>8}] 정확도 {correct / len(samples):.3f} | macro 재현율 {np.mean(list(recalls.values())):.3f} "
              f"| 평균 1위 확률 {np.mean(top_probs) * 100:.1f}% | {np.mean(times):.2f}s/장")
        print("           " + "  ".join(f"{c} {r:.2f}" for c, r in recalls.items()))

    with open("patch_eval_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\n이미지별 결과 → patch_eval_results.csv")


if __name__ == "__main__":
    main()
