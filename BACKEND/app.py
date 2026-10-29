"""
FYP backend: serves the website and runs the trained consistency model locally.

Run from the FYP folder:
    python3 BACKEND/app.py                                     # picks the model automatically
    python3 BACKEND/app.py --model results_run4_final/model.pt  # or choose one
Then open  http://127.0.0.1:8000
"""

import argparse
import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
from model import ConsistencyModel, encode, preprocess, tokenize  # noqa: E402

MAX_UPLOAD = 5 * 1024 * 1024
DOMAIN_ORDER = ["cars", "dogs", "cats", "flowers", "objects", "coco"]
DOMAIN_INFO = {
    "cars": ("Cars", "Works best when the car fills most of the photo."),
    "dogs": ("Dogs", "Works best when the dog fills most of the photo. Photos with people are not supported."),
    "cats": ("Cats", "One cat, clearly visible. Photos with people are not supported."),
    "flowers": ("Flowers", "A close-up of one kind of flower works best."),
    "objects": ("Shapes", "One coloured shape on a plain light background."),
    "coco": ("Scenes", "An everyday scene with no people in it."),
}
ATTR_LABELS = {"make": "Make", "body type": "Body type", "breed": "Breed", "animal": "Cat or dog",
               "species": "Species", "shape": "Shape", "colour": "Colour", "position": "Position",
               "object": "Object in the picture"}


class Engine:
    def __init__(self, model_path):
        self.path = Path(model_path)
        ck = torch.load(self.path, map_location="cpu")
        self.vocab = ck["vocab"]
        self.threshold = float(ck.get("threshold", 0.5))
        self.epoch = ck.get("epoch")
        self.model = ConsistencyModel(len(self.vocab))
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.options = self._load_options()
        with torch.no_grad():
            self.cand_vecs = {k: self.model.encode_text(torch.tensor([encode(c, self.vocab) for c in caps]))
                              for k, caps in self.options.items()}

    def known(self, text):
        return all(w in self.vocab for w in tokenize(text))

    def _load_options(self):
        """Every true claim seen in training, grouped by (domain, attribute), that this model can read."""
        opts = defaultdict(set)
        train = ROOT / "processed_data" / "train.csv"
        if train.exists():
            with open(train, newline="") as f:
                for r in csv.DictReader(f):
                    if r["label"] == "1" and r["attribute"] != "scene":
                        opts[(r["domain"], r["attribute"])].add(r["caption"])
        animal = set().union(*[v for (d, a), v in opts.items() if a == "animal"]) if opts else set()
        for k in opts:
            if k[1] == "animal":
                opts[k] = set(animal)  # cats and dogs both offer "cat" and "dog"
        out = {}
        for k, caps in opts.items():
            readable = sorted(c for c in caps if self.known(c))
            if len(readable) >= 2:
                out[k] = readable
        return out

    def _find_set(self, caption, domain, attribute):
        if domain and attribute and caption in self.options.get((domain, attribute), []):
            return (domain, attribute)
        hits = [k for k, caps in self.options.items() if caption in caps]
        same = [k for k in hits if k[0] == domain]
        return (same or hits or [None])[0]

    def check(self, image, caption, domain=None, attribute=None):
        x = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            img = self.model.encode_image(x)
            txt = self.model.encode_text(torch.tensor([encode(caption, self.vocab)]))
            score = torch.sigmoid(self.model.pair_logit(img, txt)).item()
            ranking, key = None, self._find_set(caption, domain, attribute)
            if key:
                caps = self.options[key]
                vals = torch.sigmoid(self.model.pair_logit(img.expand(len(caps), -1), self.cand_vecs[key])).tolist()
                ranked = sorted(zip(caps, vals), key=lambda cv: -cv[1])
                claim_rank = next(i for i, (c, _) in enumerate(ranked) if c == caption) + 1
                ranking = {"total": len(ranked), "claim_rank": claim_rank,
                           "top": [{"caption": c, "score": round(v, 4), "is_claim": c == caption}
                                   for c, v in ranked[:5]]}
        return score, ranking


app = FastAPI(title="FYP image-caption consistency detector")
ENGINE = None
LOAD_ERROR = "Model not loaded yet."


def need_engine():
    if ENGINE is None:
        raise HTTPException(503, LOAD_ERROR)
    return ENGINE


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/api/health")
def health():
    return {"model_ready": ENGINE is not None, "error": None if ENGINE else LOAD_ERROR,
            "model_path": str(ENGINE.path.relative_to(ROOT)) if ENGINE else None,
            "epoch": ENGINE.epoch if ENGINE else None}


@app.get("/api/options")
def options():
    eng = need_engine()
    by_domain = defaultdict(list)
    for (d, a), caps in sorted(eng.options.items()):
        by_domain[d].append({"id": a, "name": ATTR_LABELS.get(a, a.title()), "claims": caps})
    return {"threshold": eng.threshold,
            "domains": [{"id": d, "label": DOMAIN_INFO.get(d, (d.title(), ""))[0],
                         "tip": DOMAIN_INFO.get(d, ("", ""))[1], "attributes": by_domain[d]}
                        for d in DOMAIN_ORDER if d in by_domain]}


@app.post("/api/predict")
async def predict(image: UploadFile = File(...), caption: str = Form(...),
                  domain: str = Form(""), attribute: str = Form("")):
    eng = need_engine()
    caption = " ".join(caption.split())
    if not caption or len(caption) > 200:
        raise HTTPException(400, "Write a claim of up to 200 characters.")
    data = await image.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Choose a photo smaller than 5 MB.")
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        im = ImageOps.exif_transpose(im)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "That file isn't a photo we can read. Use JPG or PNG.")
    unknown = sorted({w for w in tokenize(caption) if w not in eng.vocab})
    score, ranking = eng.check(im, caption, domain or None, attribute or None)
    return {"caption": caption, "score": round(score, 4), "threshold": eng.threshold,
            "prediction": "consistent" if score >= eng.threshold else "inconsistent",
            "unknown_words": unknown, "ranking": ranking}


@app.get("/api/results")
def results():
    eng = need_engine()
    out = {"model_path": str(eng.path.relative_to(ROOT)), "custom": None, "by_claim": [], "clip": None}
    folder = eng.path.parent
    if (folder / "metrics.json").exists():
        m = json.loads((folder / "metrics.json").read_text())
        out["custom"] = {"test_accuracy": m.get("test_accuracy"), "best_epoch": m.get("best_epoch"),
                         "by_domain": m.get("test_accuracy_by_domain", {})}
    if (folder / "test_predictions.csv").exists():
        stats = defaultdict(lambda: [0, 0])
        with open(folder / "test_predictions.csv", newline="") as f:
            for r in csv.DictReader(f):
                k = (r["domain"], r["attribute"])
                stats[k][0] += int(r["prediction"]) == int(r["label"])
                stats[k][1] += 1
        out["by_claim"] = [{"domain": d, "attribute": ATTR_LABELS.get(a, a), "accuracy": round(c / n, 4), "pairs": n}
                           for (d, a), (c, n) in sorted(stats.items())]
    clip_file = ROOT / "results_clip" / "metrics.json"
    if clip_file.exists():
        out["clip"] = json.loads(clip_file.read_text()).get("test")
    return out


def pick_model(arg):
    if arg:
        p = Path(arg)
        return p if p.is_absolute() else ROOT / p
    for candidate in ("results_run4_final/model.pt", "results/model.pt"):
        if (ROOT / candidate).exists():
            return ROOT / candidate
    return ROOT / "results/model.pt"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="checkpoint to load, e.g. results_run4_final/model.pt")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    path = pick_model(args.model)
    try:
        ENGINE = Engine(path)
        print(f"Loaded {path.relative_to(ROOT)} (epoch {ENGINE.epoch}); "
              f"{len(ENGINE.options)} claim groups available")
    except Exception as e:
        LOAD_ERROR = f"Could not load {path}: {e}"
        print(LOAD_ERROR)
    import uvicorn
    print("Open http://127.0.0.1:%d in your browser" % args.port)
    uvicorn.run(app, host="127.0.0.1", port=args.port)
