"""
Baseline: frozen, pretrained CLIP on the SAME test pairs as the custom model.

CLIP (openai/clip-vit-base-patch32) is not trained on our data at all ("zero-shot").
Its score for a pair is the cosine similarity of its image and caption embeddings.
The cut-off between consistent / inconsistent is chosen on the VALIDATION split only,
then applied once to the test split - the same rule as for the custom model.

Run from the FYP folder (first run downloads about 600 MB):
    python3 evaluation/evaluate_clip.py
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"
CSV_DIR = ROOT / "processed_data"
CUSTOM = ROOT / "results_pretrained_224"
OUT = ROOT / "results_clip"
MODEL_ID = "openai/clip-vit-base-patch32"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def letterbox(im, fill=(128, 128, 128)):
    """Pad to a square instead of cropping, so CLIP sees the whole photo (left/right claims)."""
    im = im.convert("RGB")
    w, h = im.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    return canvas


def roc_auc(scores, labels):
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    pairs = sorted(zip(scores, labels))
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        rank_sum += avg_rank * sum(1 for k in range(i, j + 1) if pairs[k][1] == 1)
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def best_threshold(scores, labels):
    """Cut-off with the highest accuracy (score >= cut-off means consistent)."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    n, pos = len(scores), sum(labels)
    neg = n - pos
    tp = fp = 0
    best_acc, best_t = neg / n, max(scores) + 1.0
    for k, i in enumerate(order):
        if labels[i]:
            tp += 1
        else:
            fp += 1
        if k + 1 < n and scores[order[k + 1]] == scores[i]:
            continue
        acc = (tp + neg - fp) / n
        if acc > best_acc:
            nxt = scores[order[k + 1]] if k + 1 < n else scores[i] - 1e-6
            best_acc, best_t = acc, (scores[i] + nxt) / 2
    return best_t, best_acc


def paired_accuracy(rows, scores):
    """Per (photo, attribute): is the true claim scored above the false one?"""
    groups = defaultdict(lambda: {1: [], 0: []})
    for r, s in zip(rows, scores):
        groups[(r["image_path"], r["attribute"])][int(r["label"])].append(s)
    wins = total = 0
    for g in groups.values():
        for t in g[1]:
            for f in g[0]:
                wins += 1.0 if t > f else 0.5 if t == f else 0.0
                total += 1
    return wins / total if total else float("nan")


def summarise(rows, scores, threshold):
    by_dom = defaultdict(list)
    for i, r in enumerate(rows):
        by_dom[r["domain"]].append(i)
    by_dom["ALL"] = list(range(len(rows)))
    out = {}
    for d, idx in by_dom.items():
        sub_rows = [rows[i] for i in idx]
        sub_scores = [scores[i] for i in idx]
        labels = [int(r["label"]) for r in sub_rows]
        acc = sum((s >= threshold) == (l == 1) for s, l in zip(sub_scores, labels)) / len(idx)
        out[d] = {"accuracy": round(acc, 4), "roc_auc": round(roc_auc(sub_scores, labels), 4),
                  "paired_accuracy": round(paired_accuracy(sub_rows, sub_scores), 4), "pairs": len(idx)}
    return out


def clip_scores(rows, model, proc, device, img_cache, txt_cache, batch=64):
    import torch
    import torch.nn.functional as F

    def as_tensor(x, key):
        if torch.is_tensor(x):
            return x
        v = getattr(x, key, None)
        return v if v is not None else x.pooler_output

    new_imgs = sorted({r["image_path"] for r in rows} - set(img_cache))
    for b in range(0, len(new_imgs), batch):
        paths = new_imgs[b:b + batch]
        ims = []
        for p in paths:
            with Image.open(DATA / p) as im:
                ims.append(letterbox(im))
        inputs = proc(images=ims, return_tensors="pt").to(device)
        with torch.no_grad():
            e = F.normalize(as_tensor(model.get_image_features(**inputs), "image_embeds"), dim=-1).cpu()
        img_cache.update(zip(paths, e))
        print(f"    images {min(b + batch, len(new_imgs))}/{len(new_imgs)}", end="\r")
    print()

    new_txt = sorted({r["caption"] for r in rows} - set(txt_cache))
    for b in range(0, len(new_txt), batch):
        caps = new_txt[b:b + batch]
        inputs = proc(text=caps, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            e = F.normalize(as_tensor(model.get_text_features(**inputs), "text_embeds"), dim=-1).cpu()
        txt_cache.update(zip(caps, e))

    return [float(img_cache[r["image_path"]] @ txt_cache[r["caption"]]) for r in rows]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto", help="auto, cpu or mps")
    args = p.parse_args()

    import torch
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        raise SystemExit("CLIP needs the transformers library. Run:  pip3 install transformers")

    device = (("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else args.device)
    val_rows, test_rows = read_csv(CSV_DIR / "val.csv"), read_csv(CSV_DIR / "test.csv")
    print(f"Device: {device}   val pairs {len(val_rows)}   test pairs {len(test_rows)}")
    print(f"Loading {MODEL_ID} (first run downloads about 600 MB)...")
    model = CLIPModel.from_pretrained(MODEL_ID).eval().to(device)
    proc = CLIPProcessor.from_pretrained(MODEL_ID)

    img_cache, txt_cache = {}, {}
    print("  scoring validation pairs...")
    val_scores = clip_scores(val_rows, model, proc, device, img_cache, txt_cache)
    threshold, val_acc = best_threshold(val_scores, [int(r["label"]) for r in val_rows])
    print(f"  cut-off chosen on validation: {threshold:.4f} (val accuracy {val_acc:.3f})")
    print("  scoring test pairs...")
    test_scores = clip_scores(test_rows, model, proc, device, img_cache, txt_cache)

    clip = summarise(test_rows, test_scores, threshold)
    OUT.mkdir(exist_ok=True)
    with open(OUT / "test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "caption", "domain", "attribute", "label", "score", "prediction"])
        for r, s in zip(test_rows, test_scores):
            w.writerow([r["image_path"], r["caption"], r["domain"], r["attribute"], r["label"],
                        f"{s:.4f}", int(s >= threshold)])
    with open(OUT / "metrics.json", "w") as f:
        json.dump({"model": MODEL_ID, "zero_shot": True, "threshold_from_validation": threshold,
                   "test": clip}, f, indent=2)

    custom = None
    pred_file = CUSTOM / "test_predictions.csv"
    if pred_file.exists():
        preds = read_csv(pred_file)
        key = lambda r: (r["image_path"], r["caption"], r["label"])
        if sorted(map(key, preds)) == sorted(map(key, test_rows)):
            score_of = {key(r): float(r["score"]) for r in preds}
            custom = summarise(test_rows, [score_of[key(r)] for r in test_rows], 0.5)
        else:
            print("\n  NOTE: the custom model's predictions were made from a different test set.")

    print("\n" + "=" * 78)
    print(f"{'domain':10}  {'accuracy':>17}  {'ROC-AUC':>17}  {'paired accuracy':>21}")
    print(f"{'':10}  {'custom':>8} {'CLIP':>8}  {'custom':>8} {'CLIP':>8}  {'custom':>10} {'CLIP':>10}")
    for d in sorted(clip, key=lambda d: (d == "ALL", d)):
        c = custom.get(d) if custom else None
        f = lambda m: f"{c[m]:.3f}" if c else "-"
        print(f"{d:10}  {f('accuracy'):>8} {clip[d]['accuracy']:>8.3f}  {f('roc_auc'):>8} "
              f"{clip[d]['roc_auc']:>8.3f}  {f('paired_accuracy'):>10} {clip[d]['paired_accuracy']:>10.3f}")
    print("=" * 78)
    print("50% / 0.5 = guessing.  Saved results_clip/metrics.json and results_clip/test_predictions.csv")


if __name__ == "__main__":
    main()
