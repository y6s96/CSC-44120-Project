import csv, json, math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "results_pretrained_224" / "test_predictions.csv"
SCRATCH = ROOT / "results_run7_scratch" / "test_predictions.csv"
OUT = ROOT / "results_analysis"


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def prf(rows):
    tp = sum(1 for r in rows if r["label"] == "1" and r["prediction"] == "1")
    fp = sum(1 for r in rows if r["label"] == "0" and r["prediction"] == "1")
    fn = sum(1 for r in rows if r["label"] == "1" and r["prediction"] == "0")
    tn = len(rows) - tp - fp - fn
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    return {"n": len(rows), "accuracy": round((tp + tn) / len(rows), 4), "precision": round(p, 4),
            "recall": round(rc, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def value_of(caption):
    words = caption.rstrip(".").split()
    for cut in ("by", "a", "an", "is"):
        if cut in words:
            i = len(words) - 1 - words[::-1].index(cut)
            return " ".join(words[i + 1:])
    return caption


def mcnemar(a_right, b_right):
    b = sum(1 for x, y in zip(a_right, b_right) if x and not y)
    c = sum(1 for x, y in zip(a_right, b_right) if y and not x)
    if b + c == 0:
        return b, c, 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = math.erfc(math.sqrt(chi2 / 2))
    return b, c, round(chi2, 2), p


def main():
    rows = read(FINAL)
    out = {}
    print("1. FINAL MODEL: precision / recall / F1")
    out["by_domain"] = {d: prf([r for r in rows if r["domain"] == d]) for d in sorted({r["domain"] for r in rows})}
    out["by_domain"]["ALL"] = prf(rows)
    for d, m in out["by_domain"].items():
        print(f"  {d:9} acc {m['accuracy']:.3f}  P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}"
              f"   (TP {m['tp']} FP {m['fp']} FN {m['fn']} TN {m['tn']})")
    claims = sorted({(r["domain"], r["attribute"]) for r in rows})
    out["by_claim"] = {f"{d}/{a}": prf([r for r in rows if r["domain"] == d and r["attribute"] == a]) for d, a in claims}
    for k, m in out["by_claim"].items():
        print(f"  {k:22} P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}")

    print("\n2. WHERE THE ERRORS ARE (final model)")
    out["errors"] = {}
    for d, a in claims:
        sub = [r for r in rows if r["domain"] == d and r["attribute"] == a]
        fa = Counter(value_of(r["caption"]) for r in sub if r["label"] == "0" and r["prediction"] == "1")
        fr = Counter(value_of(r["caption"]) for r in sub if r["label"] == "1" and r["prediction"] == "0")
        out["errors"][f"{d}/{a}"] = {"wrongly_accepted": fa.most_common(5), "wrongly_rejected": fr.most_common(5)}
        if fa or fr:
            print(f"  {d}/{a}")
            print(f"     false claims accepted most often: {fa.most_common(5)}")
            print(f"     true claims rejected most often:  {fr.most_common(5)}")

    print("\n3. CONFIDENCE (final model)")
    right = [float(r["score"]) for r in rows if r["label"] == r["prediction"]]
    wrong = [float(r["score"]) for r in rows if r["label"] != r["prediction"]]
    margin = lambda s: abs(s - 0.5)
    conf = lambda xs: sum(1 for s in xs if margin(s) > 0.4) / len(xs) if xs else 0
    out["confidence"] = {"right": len(right), "wrong": len(wrong),
                         "share_right_confident": round(conf(right), 4), "share_wrong_confident": round(conf(wrong), 4)}
    print(f"  right answers: {len(right)}, of which {conf(right):.1%} with score < 0.1 or > 0.9")
    print(f"  wrong answers: {len(wrong)}, of which {conf(wrong):.1%} with score < 0.1 or > 0.9")

    print("\n4. McNEMAR TEST: pretrained (final) vs scratch (run 7) on shared test pairs")
    if SCRATCH.exists():
        key = lambda r: (r["image_path"], r["caption"], r["label"])
        s_rows = {key(r): r for r in read(SCRATCH)}
        out["mcnemar"] = {}
        for d in sorted({r["domain"] for r in rows}):
            shared = [(r, s_rows[key(r)]) for r in rows if r["domain"] == d and key(r) in s_rows]
            if not shared:
                print(f"  {d:9} no shared test pairs")
                continue
            a = [f["label"] == f["prediction"] for f, _ in shared]
            b = [s["label"] == s["prediction"] for _, s in shared]
            only_a, only_b, chi2, p = mcnemar(a, b)
            out["mcnemar"][d] = {"shared_pairs": len(shared), "final_acc": round(sum(a) / len(a), 4),
                                 "scratch_acc": round(sum(b) / len(b), 4), "only_final_right": only_a,
                                 "only_scratch_right": only_b, "chi2": chi2, "p_value": p}
            print(f"  {d:9} shared {len(shared):5}  final {sum(a)/len(a):.3f}  scratch {sum(b)/len(b):.3f}"
                  f"  only-final-right {only_a:4}  only-scratch-right {only_b:4}  chi2 {chi2:7.2f}  p {p:.2e}")
    else:
        print("  results_run7_scratch/test_predictions.csv not found")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results_analysis/analysis.json")


if __name__ == "__main__":
    main()
