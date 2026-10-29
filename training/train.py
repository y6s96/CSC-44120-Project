"""
Train the consistency model on processed_data/*.csv and evaluate on the test split.
  scratch model (main):       python3 training/train.py --epochs 30
  pretrained-image variant:   python3 training/train.py --backbone resnet18 --epochs 10 --out results_pretrained
"""

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import ConsistencyModel, encode, preprocess, tokenize, MAX_TOKENS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"
CSV_DIR = ROOT / "processed_data"


def read_rows(split):
    with open(CSV_DIR / f"{split}.csv", newline="") as f:
        return list(csv.DictReader(f))


def build_vocab(rows):
    counts = Counter(w for r in rows for w in tokenize(r["caption"]))
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counts.most_common():
        vocab[word] = len(vocab)
    return vocab


def cand_key(r):
    return ("cats+dogs", "animal") if r["attribute"] == "animal" else (r["domain"], r["attribute"])


def build_candidates(train_rows):
    cands = defaultdict(set)
    for r in train_rows:
        if r["attribute"] != "scene" and r["label"] == "1":
            cands[cand_key(r)].add(r["caption"])
    keys = sorted(cands)
    return keys, {k: sorted(cands[k]) for k in keys}


def random_crop(im, min_scale=0.85):
    w, h = im.size
    s = random.uniform(min_scale, 1.0)
    cw, ch = max(1, int(w * s)), max(1, int(h * s))
    x, y = random.randint(0, w - cw), random.randint(0, h - ch)
    return im.crop((x, y, x + cw, y + ch))


class PairDataset(Dataset):
    def __init__(self, rows, vocab, size, imagenet, keys=None, cands=None, augment=False):
        self.rows, self.vocab, self.augment = rows, vocab, augment
        self.size, self.imagenet = size, imagenet
        self.aux = []
        true_caption = {(r["image_path"], r["attribute"]): r["caption"]
                        for r in rows if r["label"] == "1"}
        key_id = {k: i for i, k in enumerate(keys or [])}
        for r in rows:
            k = cand_key(r)
            truth = true_caption.get((r["image_path"], r["attribute"]))
            if cands and k in key_id and truth in cands[k]:
                self.aux.append((key_id[k], cands[k].index(truth)))
            else:
                self.aux.append((-1, -1))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(DATA / r["image_path"]) as im:
            if self.augment and r["domain"] not in ("objects", "coco"):
                im = random_crop(im)
            image = preprocess(im, self.size, self.imagenet)
        tokens = torch.tensor(encode(r["caption"], self.vocab), dtype=torch.long)
        key, target = self.aux[i]
        return image, tokens, torch.tensor(float(r["label"])), i, key, target


def run_epoch(model, loader, device, cand_tokens=None, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, aux_total, aux_n, scores, labels, indices = 0.0, 0.0, 0, [], [], []
    with torch.set_grad_enabled(training):
        for images, tokens, y, idx, keys, targets in loader:
            images, tokens, y = images.to(device), tokens.to(device), y.to(device)
            keys, targets = keys.to(device), targets.to(device)
            img = model.encode_image(images)
            logits = model.pair_logit(img, model.encode_text(tokens))
            loss = F.binary_cross_entropy_with_logits(logits, y)
            total += loss.item() * len(y)
            if training and cand_tokens:
                aux_sum, n = 0.0, 0
                for k in keys.unique().tolist():
                    sel = (keys == k) & (targets >= 0)
                    if k < 0 or not sel.any():
                        continue
                    cand_vec = model.encode_text(cand_tokens[k])
                    aux_logits = model.scale * img[sel] @ cand_vec.T
                    aux_sum = aux_sum + F.cross_entropy(aux_logits, targets[sel], reduction="sum")
                    n += int(sel.sum())
                if n:
                    aux_loss = aux_sum / n
                    loss = loss + aux_loss
                    aux_total += aux_loss.item() * n
                    aux_n += n
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scores += torch.sigmoid(logits).detach().cpu().tolist()
            labels += y.cpu().tolist()
            indices += idx.tolist()
    acc = float(np.mean((np.array(scores) >= 0.5) == (np.array(labels) == 1)))
    return total / len(labels), acc, scores, indices, (aux_total / aux_n if aux_n else 0.0)


def per_domain_accuracy(rows, scores, indices):
    stats = defaultdict(lambda: [0, 0])
    for score, i in zip(scores, indices):
        stats[rows[i]["domain"]][0] += int(int(score >= 0.5) == int(rows[i]["label"]))
        stats[rows[i]["domain"]][1] += 1
    return {d: round(c / n, 4) for d, (c, n) in sorted(stats.items())}


def per_claim_accuracy(rows, scores, indices):
    stats = defaultdict(lambda: [0, 0])
    for score, i in zip(scores, indices):
        k = f"{rows[i]['domain']}/{rows[i]['attribute']}"
        stats[k][0] += int(int(score >= 0.5) == int(rows[i]["label"]))
        stats[k][1] += 1
    return {k: round(c / n, 4) for k, (c, n) in sorted(stats.items())}


def pick_one_accuracy(model, rows, cands, vocab, device, size, imagenet, batch=64):
    """For each test photo: is its TRUE claim the top choice out of ALL claims of that kind?"""
    model.eval()
    todo = [r for r in rows if r["label"] == "1" and r["attribute"] != "scene"
            and cand_key(r) in cands and r["caption"] in cands[cand_key(r)]]
    paths = sorted({r["image_path"] for r in todo})
    img_vec = {}
    with torch.no_grad():
        for b in range(0, len(paths), batch):
            chunk = paths[b:b + batch]
            ims = []
            for p in chunk:
                with Image.open(DATA / p) as im:
                    ims.append(preprocess(im, size, imagenet))
            vecs = model.encode_image(torch.stack(ims).to(device)).cpu()
            img_vec.update(zip(chunk, vecs))
        cand_vec = {k: model.encode_text(torch.tensor([encode(c, vocab) for c in caps],
                                                      dtype=torch.long).to(device)).cpu()
                    for k, caps in cands.items()}
    stats = defaultdict(lambda: [0, 0, 0])
    for r in todo:
        k = cand_key(r)
        best = int((cand_vec[k] @ img_vec[r["image_path"]]).argmax())
        s = stats[f"{r['domain']}/{r['attribute']}"]
        s[0] += int(cands[k][best] == r["caption"])
        s[1] += 1
        s[2] = len(cands[k])
    return {k: {"accuracy": round(c / n, 4), "photos": n, "options": o, "chance": round(1 / o, 4)}
            for k, (c, n, o) in sorted(stats.items())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone", default="small", choices=["small", "resnet18"])
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--out", default="results")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    pretrained = args.backbone == "resnet18"
    size = args.image_size or (160 if pretrained else 128)
    results = ROOT / args.out

    train_rows, val_rows, test_rows = read_rows("train"), read_rows("val"), read_rows("test")
    print(f"Device: {device}   backbone: {args.backbone}   image size: {size}   saving to: {args.out}/")
    print(f"Pairs  train {len(train_rows)}  val {len(val_rows)}  test {len(test_rows)}")
    print("Domains:", dict(Counter(r["domain"] for r in train_rows)))

    vocab = build_vocab(train_rows)
    keys, cands = build_candidates(train_rows)
    print("Auxiliary caption sets:", {f"{d}/{a}": len(cands[(d, a)]) for d, a in keys})
    cand_tokens = [torch.tensor([encode(c, vocab) for c in cands[k]], dtype=torch.long).to(device)
                   for k in keys]

    loaders = {
        "train": DataLoader(PairDataset(train_rows, vocab, size, pretrained, keys, cands, augment=True),
                            batch_size=args.batch_size, shuffle=True, num_workers=args.workers),
        "val": DataLoader(PairDataset(val_rows, vocab, size, pretrained),
                          batch_size=args.batch_size, num_workers=args.workers),
        "test": DataLoader(PairDataset(test_rows, vocab, size, pretrained),
                           batch_size=args.batch_size, num_workers=args.workers),
    }

    model = ConsistencyModel(len(vocab), backbone=args.backbone, pretrained=True).to(device)
    if pretrained:
        base = [p for n, p in model.named_parameters() if n.startswith("image_encoder.") and ".fc." not in n]
        base_ids = {id(p) for p in base}
        rest = [p for p in model.parameters() if id(p) not in base_ids]
        optimizer = torch.optim.Adam([{"params": base, "lr": args.lr / 10}, {"params": rest, "lr": args.lr}],
                                     weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    results.mkdir(exist_ok=True)

    best_val, history = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        tr_loss, tr_acc, _, _, aux = run_epoch(model, loaders["train"], device, cand_tokens, optimizer)
        va_loss, va_acc, va_scores, va_idx, _ = run_epoch(model, loaders["val"], device)
        scheduler.step()
        va_dom = per_domain_accuracy(val_rows, va_scores, va_idx)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "aux_loss": aux,
                        "val_loss": va_loss, "val_acc": va_acc, "val_acc_by_domain": va_dom})
        mark = ""
        if va_loss < best_val:
            best_val, mark = va_loss, "  <- best, saved"
            torch.save({"state_dict": model.state_dict(), "vocab": vocab, "epoch": epoch,
                        "backbone": args.backbone, "image_size": size, "imagenet_norm": pretrained,
                        "max_tokens": MAX_TOKENS, "threshold": 0.5, "model_type": "dual_encoder_v3"},
                       results / "model.pt")
        print(f"Epoch {epoch:2d}  train loss {tr_loss:.3f} acc {tr_acc:.3f} aux {aux:.3f}  |  "
              f"val loss {va_loss:.3f} acc {va_acc:.3f}  ({time.time() - start:.0f}s){mark}")
        print("          val by domain: " + "  ".join(f"{d} {a:.2f}" for d, a in va_dom.items()))

    checkpoint = torch.load(results / "model.pt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    _, te_acc, scores, indices, _ = run_epoch(model, loaders["test"], device)
    domain_acc = per_domain_accuracy(test_rows, scores, indices)
    claim_acc = per_claim_accuracy(test_rows, scores, indices)
    pick_one = pick_one_accuracy(model, test_rows, cands, vocab, device, size, pretrained)

    with open(results / "test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "caption", "domain", "attribute", "label", "score", "prediction"])
        for score, i in zip(scores, indices):
            r = test_rows[i]
            w.writerow([r["image_path"], r["caption"], r["domain"], r["attribute"],
                        r["label"], f"{score:.4f}", int(score >= 0.5)])
    with open(results / "metrics.json", "w") as f:
        json.dump({"backbone": args.backbone, "image_size": size, "best_epoch": checkpoint["epoch"],
                   "test_accuracy": round(te_acc, 4), "test_accuracy_by_domain": domain_acc,
                   "test_accuracy_by_claim": claim_acc, "pick_one_accuracy": pick_one,
                   "history": history}, f, indent=2)

    print(f"\nBest epoch: {checkpoint['epoch']}")
    print(f"TEST accuracy: {te_acc:.3f}  (50% = guessing)")
    for d, a in domain_acc.items():
        print(f"  {d:10} {a:.3f}")
    print("\nPick the right claim out of ALL options (harder):")
    for k, v in pick_one.items():
        print(f"  {k:20} {v['accuracy']:.3f}   ({v['options']} options, chance {v['chance']:.3f})")
    print(f"\nSaved {args.out}/model.pt, metrics.json, test_predictions.csv")


if __name__ == "__main__":
    main()
