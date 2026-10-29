"""
Build image-caption pairs for the FYP project from REAL dataset labels.
Run from the FYP folder:  python3 process_all_datasets.py [--skip-coco] [--per-domain 1500]
"""

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent      # the FYP folder
DATA = ROOT / "Data"
OUT = ROOT / "processed_data"
SEED = 42
IMG_EXT = {".jpg", ".jpeg", ".png"}
PEOPLE = None  # person filter, set up in main()

TEMPLATES = {
    ("cars", "make"): "The car is made by {v}.",
    ("cars", "body type"): "The car is {a} {v}.",
    ("cars", "colour"): "The car is {v}.",
    ("dogs", "breed"): "The dog is {a} {v}.",
    ("dogs", "animal"): "The animal is {a} {v}.",
    ("cats", "animal"): "The animal is {a} {v}.",
    ("flowers", "species"): "The flower is {a} {v}.",
    ("objects", "shape"): "The shape is {a} {v}.",
    ("objects", "colour"): "The shape is {v}.",
    ("objects", "position"): "The shape is on the {v}.",
    ("coco", "object"): "There is {a} {v} in the picture.",
}
FIXED_VOCAB = {"animal": ["cat", "dog"]}


def article(word):
    if word.upper() == "SUV" or word[0].lower() in "aeiou":
        return "an"
    return "a"


def caption_for(domain, attribute, value):
    return TEMPLATES[(domain, attribute)].format(a=article(value), v=value)


def images_under(folder):
    return sorted(p for p in folder.rglob("*")
                  if p.suffix.lower() in IMG_EXT and "venv" not in p.parts)


def readable(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def sample_readable(records, n, rng, name):
    rng.shuffle(records)
    kept, broken, people = [], 0, 0
    if PEOPLE is not None and name != "objects":
        print(f"  {name}: checking photos for people...")
    for r in records:
        if len(kept) >= n:
            break
        if not readable(r["image"]):
            broken += 1
        elif PEOPLE is not None and name != "objects" and PEOPLE.has_person(r["image"]):
            people += 1
        else:
            kept.append(r)
    note = f" (skipped {broken} broken files)" if broken else ""
    if people:
        note += f" (removed {people} photos with people)"
    print(f"  {name}: using {len(kept)} images{note}")
    return kept


def find_one(pattern):
    hits = [p for p in DATA.rglob(pattern) if "venv" not in p.parts]
    return hits[0] if hits else None


CAR_BODY = [("Convertible", "convertible"), ("Coupe", "coupe"), ("Sedan", "sedan"),
            ("Hatchback", "hatchback"), ("Wagon", "estate car"), ("Minivan", "van"),
            ("Van", "van"), ("SUV", "SUV"), ("Cab", "pickup truck")]
CAR_MULTIWORD_MAKES = ["AM General", "Aston Martin", "Land Rover"]
YEAR_AT_END = re.compile(r"(19|20)\d\d$")


def parse_car_class(name):
    make = next((m for m in CAR_MULTIWORD_MAKES if name.startswith(m + " ")), name.split()[0])
    words = name.split()
    body = next((v for k, v in CAR_BODY if k in words), None)
    return make, body


CROPS = DATA / "_crops"


def square_crop(image_path, box, margin=0.1, size=256):
    """Cut out the labelled dog/car plus a 10% margin on each side (no padding), saved in Data/_crops/."""
    dest = (CROPS / Path(image_path).relative_to(DATA)).with_suffix(".jpg")
    if dest.exists():
        return dest
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        W, H = im.size
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            raise ValueError("box too small")
        mx, my = (x2 - x1) * margin, (y2 - y1) * margin
        crop = im.crop((int(max(0, x1 - mx)), int(max(0, y1 - my)),
                        int(min(W, x2 + mx)), int(min(H, y2 + my))))
        crop.thumbnail((size, size))
        dest.parent.mkdir(parents=True, exist_ok=True)
        crop.save(dest, quality=95)
    return dest


def apply_crops(records, name, thing):
    done = 0
    for r in records:
        if r.get("box"):
            try:
                r["image"] = square_crop(r["image"], r["box"])
                done += 1
            except Exception:
                pass  # keep the full photo
    print(f"    {name}: cropped {done}/{len(records)} photos to the labelled {thing}")
    return records


def car_colour(crop_path):
    """Paint colour from the middle of the car body (below the windows). None if no colour covers half."""
    import colorsys
    with Image.open(crop_path) as im:
        im = im.convert("RGB").resize((64, 64))
    counts = Counter()
    for r, g, b in im.crop((16, 29, 48, 48)).getdata():
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        deg = h * 360
        if v < 0.22:
            c = "black"
        elif s < 0.20:
            c = "white" if v > 0.72 else "grey"
        elif deg < 20 or deg >= 330:
            c = "red"
        elif deg < 45:
            c = "orange"
        elif deg < 70:
            c = "yellow"
        elif deg < 170:
            c = "green"
        elif deg < 260:
            c = "blue"
        else:
            c = "other"
        counts[c] += 1
    label, n = counts.most_common(1)[0]
    return label if label != "other" and n >= 0.5 * sum(counts.values()) else None


def add_car_colours(records):
    done, by_colour = 0, Counter()
    for r in records:
        if r.get("box") and "_crops" in Path(r["image"]).parts:
            c = car_colour(r["image"])
            if c:
                r["attrs"]["colour"] = c
                by_colour[c] += 1
                done += 1
    print(f"    cars: colour read for {done}/{len(records)} cars: " +
          ", ".join(f"{c} {n}" for c, n in by_colour.most_common()))
    try:
        from PIL import ImageDraw
        sample = [r for r in records if "colour" in r["attrs"]]
        sample = random.Random(7).sample(sample, min(48, len(sample)))
        sheet = Image.new("RGB", (8 * 160, 6 * 150), "white")
        draw = ImageDraw.Draw(sheet)
        for i, r in enumerate(sample):
            with Image.open(r["image"]) as im:
                im = im.convert("RGB")
                im.thumbnail((150, 120))
            x, y = (i % 8) * 160 + 5, (i // 8) * 150 + 5
            sheet.paste(im, (x, y))
            draw.text((x, y + 124), f"{i + 1}: {r['attrs']['colour']}", fill="black")
        OUT.mkdir(exist_ok=True)
        sheet.save(OUT / "car_colour_check.jpg", quality=90)
        print("    cars: check sheet saved to processed_data/car_colour_check.jpg")
    except Exception as e:
        print(f"    cars: could not make the check sheet ({e})")
    return records


def load_car_boxes(folder, names):
    """anno_train.csv / anno_test.csv rows: filename, x1, y1, x2, y2, class number (1-based)."""
    boxes = {}
    for split in ("train", "test"):
        anno = folder / f"anno_{split}.csv"
        if not anno.exists():
            continue
        with open(anno, newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 6 and row[5].strip().isdigit():
                    idx = int(row[5]) - 1
                    cls = names[idx] if 0 <= idx < len(names) else None
                    try:
                        box = tuple(int(float(v)) for v in row[1:5])
                    except ValueError:
                        continue
                    boxes[(split, row[0].strip())] = (box, cls)
    return boxes


def load_cars(n, rng):
    folder = DATA / "CARS"
    if not folder.exists():
        print("  cars: Data/CARS not found, skipping")
        return []
    names = []
    if (folder / "names.csv").exists():
        with open(folder / "names.csv", newline="") as f:
            names = [row[0].strip() for row in csv.reader(f) if row]
    boxes = load_car_boxes(folder, names)
    same = lambda a, b: a.replace("/", "-").strip().lower() == b.replace("/", "-").strip().lower()

    records = []
    for p in images_under(folder):
        parts = p.relative_to(DATA).parts
        split = "train" if "train" in parts else "test" if "test" in parts else None
        info = boxes.get((split, p.name))
        cls = p.parent.name if YEAR_AT_END.search(p.parent.name) else (info[1] if info else None)
        if not cls:
            continue
        make, body = parse_car_class(cls)
        attrs = {"make": make}
        if body:
            attrs["body type"] = body
        rec = {"image": p, "domain": "cars", "attrs": attrs}
        if info and info[1] and same(info[1], cls):  # only use a box that belongs to this exact car
            rec["box"] = info[0]
        records.append(rec)
    if not records:
        print("  cars: found no labelled images in Data/CARS, skipping")
        return []
    return add_car_colours(apply_crops(sample_readable(records, n, rng, "cars"), "cars", "car"))


def load_dog_box(xml_path):
    """Stanford Dogs annotation (XML): use the largest dog box in the photo."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None
    best = None
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            x1, y1, x2, y2 = (int(float(bb.find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax"))
        except (AttributeError, ValueError):
            continue
        area = (x2 - x1) * (y2 - y1)
        if best is None or area > best[0]:
            best = (area, (x1, y1, x2, y2))
    return best[1] if best else None


def normalised_copy(image_path, size=256):
    """Same resize + JPEG step as the crops, so cat and dog photos differ only in what they show."""
    dest = (DATA / "_norm" / Path(image_path).relative_to(DATA)).with_suffix(".jpg")
    if not dest.exists():
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size))
            dest.parent.mkdir(parents=True, exist_ok=True)
            im.save(dest, quality=95)
    return dest


def load_dogs(n, rng):
    folder = DATA / "DOGS"
    if not folder.exists():
        print("  dogs: Data/DOGS not found, skipping")
        return []
    records = []
    for p in images_under(folder):
        m = re.match(r"n\d+-(.+)", p.parent.name)
        if m:
            breed = m.group(1).replace("_", " ")
            records.append({"image": p, "domain": "dogs", "attrs": {"breed": breed}})
    if not records:
        print("  dogs: no breed folders found in Data/DOGS, skipping")
        return []
    kept = sample_readable(records, n, rng, "dogs")
    ann_dir = folder / "annotations"
    ann = {f.name: f for f in ann_dir.rglob("*") if f.is_file() and f.suffix == ""} if ann_dir.exists() else {}
    for r in kept:
        r["original"] = r["image"]
        f = ann.get(r["image"].stem)
        r["box"] = load_dog_box(f) if f else None
    kept = apply_crops(kept, "dogs", "dog")
    out = []
    for r in kept:
        # breed claim: the crop.  cat-or-dog claim: the full photo, like the cats.
        out.append({"image": r["image"], "group": r["original"], "domain": "dogs",
                    "attrs": {"breed": r["attrs"]["breed"]}})
        out.append({"image": normalised_copy(r["original"]), "group": r["original"], "domain": "dogs",
                    "attrs": {"animal": "dog"}})
    return out


def is_labelled_cat(path, folder):
    """Only photos marked as cats: name starts with "cat" or inside a Cat/Cats folder."""
    name = path.name.lower()
    subfolders = [part.lower() for part in path.relative_to(folder).parts[:-1]]
    if name.startswith("dog") or any(part in {"dog", "dogs"} for part in subfolders):
        return False
    return name.startswith("cat") or any(part in {"cat", "cats"} for part in subfolders)


def load_cats(n, rng):
    folder = DATA / "CATS"
    if not folder.exists():
        print("  cats: Data/CATS not found, skipping")
        return []
    everything = images_under(folder)
    records = [{"image": p, "domain": "cats", "attrs": {"animal": "cat"}}
               for p in everything if is_labelled_cat(p, folder)]
    print(f"  cats: {len(records)} of {len(everything)} photos are labelled as cats")
    if not records:
        print("  cats: no labelled cat photos found, skipping")
        return []
    kept = sample_readable(records, n, rng, "cats")
    for r in kept:
        r["group"] = r["image"]
        r["image"] = normalised_copy(r["image"])
    return kept


FLOWER_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger",
    "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian",
    "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower",
    "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil",
    "sword lily", "poinsettia", "bolero deep blue", "wallflower", "marigold", "buttercup",
    "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
    "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
    "pink-yellow dahlia", "cautleya spicata", "japanese anemone", "black-eyed susan",
    "silverbush", "californian poppy", "osteospermum", "spring crocus", "bearded iris",
    "windflower", "tree poppy", "gazania", "azalea", "water lily", "rose", "thorn apple",
    "morning glory", "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia",
    "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss",
    "foxglove", "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia",
    "blanket flower", "trumpet creeper", "blackberry lily",
]
assert len(FLOWER_NAMES) == 102


def load_flowers(n, rng):
    files = {p.name: p for p in DATA.rglob("image_*.jpg")
             if "venv" not in p.parts and re.fullmatch(r"image_\d{5}\.jpg", p.name)}
    if not files:
        print("  flowers: no image_00001.jpg-style files found, skipping")
        return []
    mat = find_one("imagelabels.mat")
    if mat is None:
        print(f"  flowers: found {len(files)} flower images but NOT imagelabels.mat, skipping.")
        print("    Download it into the same folder as the flower images:")
        print("    https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat")
        return []
    try:
        from scipy.io import loadmat
    except ImportError:
        print("  flowers: needs scipy. Run:  pip3 install scipy")
        return []
    labels = loadmat(mat)["labels"].ravel()
    records = []
    for i, cls in enumerate(labels, start=1):
        p = files.get(f"image_{i:05d}.jpg")
        if p:
            records.append({"image": p, "domain": "flowers",
                            "attrs": {"species": FLOWER_NAMES[int(cls) - 1]}})
    return sample_readable(records, n, rng, "flowers")


PERSON_WORDS = re.compile(
    r"\b(man|men|woman|women|person|people|persons|boy|boys|girl|girls|child|children|kid|kids|"
    r"lady|ladies|guy|guys|player|players|someone|baby|babies|skier|skiers|surfer|surfers|"
    r"skateboarder|rider|riders|crowd|he|she|his|her|him|they|them|their|toddler|teen|teenager|"
    r"adult|adults|family|couple|chef|cook|driver|police|officer|soldier|human|humans|face|faces|"
    r"hand|hands|catcher|batter|pitcher|umpire|tennis|frisbee)\b", re.I)


COCO_NAMES = {"skis": "pair of skis", "scissors": "pair of scissors"}
MIN_AREA = 0.02  # an object counts as "in the picture" only if its box covers >= 2% of the photo


def load_coco(n, rng):
    """Presence claims from COCO object labels: "There is a cat in the picture."
    Any photo with a person label, a person word in its captions, or (later) a person
    found by the detector is removed."""
    if not (DATA / "COCO").exists():
        print("  coco: Data/COCO not found, skipping")
        return []
    files = {p.name: p for p in DATA.rglob("COCO_*2014_*.jpg") if "venv" not in p.parts}
    if not files:
        print("  coco: no COCO_*2014_*.jpg images found, skipping")
        return []
    records = []
    for split in ("train2014", "val2014"):
        inst_file = find_one(f"instances_{split}.json")
        if not inst_file:
            continue
        print(f"  coco {split}: loading object labels (this can take a minute)...")
        with open(inst_file) as f:
            inst = json.load(f)
        cat_name = {c["id"]: COCO_NAMES.get(c["name"], c["name"]) for c in inst["categories"]}
        size = {im["id"]: (im["width"], im["height"]) for im in inst["images"]}
        id_file = {im["id"]: im["file_name"] for im in inst["images"]}
        present, big = defaultdict(set), defaultdict(set)
        for a in inst["annotations"]:
            name = cat_name[a["category_id"]]
            present[a["image_id"]].add(name)
            w, h = size[a["image_id"]]
            if a["bbox"][2] * a["bbox"][3] >= MIN_AREA * w * h:
                big[a["image_id"]].add(name)
        del inst

        person_words = set()
        cap_file = find_one(f"captions_{split}.json")
        if cap_file:
            with open(cap_file) as f:
                for c in json.load(f)["annotations"]:
                    if PERSON_WORDS.search(c["caption"]):
                        person_words.add(c["image_id"])

        before = len(records)
        for img_id, names in present.items():
            if "person" in names or img_id in person_words or not big[img_id]:
                continue
            path = files.get(id_file.get(img_id, ""))
            if path:
                records.append({"image": path, "domain": "coco", "present": names,
                                "attrs": {"object": sorted(big[img_id])}})
        print(f"  coco {split}: {len(records) - before} photos with no people")
    if not records:
        print("  coco: nothing usable, skipping")
        return []
    for r in records:  # one claim per photo: a clearly visible object, picked at random
        r["attrs"] = {"object": rng.choice(r["attrs"]["object"])}
    return sample_readable(records, n, rng, "coco")


def load_objects(n, rng):
    ann = DATA / "synthetic_objects" / "annotations.json"
    if not ann.exists():
        print("  objects: run  python3 Data/generate_objects_dataset.py  first, skipping")
        return []
    with open(ann) as f:
        items = json.load(f)
    records = []
    for it in items:
        attrs = {"shape": it["object_type"], "colour": it["color"]}
        if "position" in it:
            attrs["position"] = it["position"]
        records.append({"image": DATA / "synthetic_objects" / "images" / it["image_name"],
                        "domain": "objects", "attrs": attrs})
    return sample_readable(records, n, rng, "objects")


def make_pairs(records, rng):
    rows = []
    pool = defaultdict(list)
    for r in records:
        for attr, val in r.get("attrs", {}).items():
            key = "animal" if attr == "animal" else (r["domain"], attr)
            pool[key].append(val)
    for r in records:
        for attr, val in r.get("attrs", {}).items():
            key = "animal" if attr == "animal" else (r["domain"], attr)
            # a wrong value must not be in the photo at all (COCO photos show several objects)
            options = set(pool[key]) - {val} - set(r.get("present", ()))
            if not options:
                continue
            wrong = val
            while wrong not in options:
                wrong = rng.choice(pool[key])
            for value, label in ((val, 1), (wrong, 0)):
                rows.append({"image": r["image"], "group": r.get("group", r["image"]), "domain": r["domain"], "attribute": attr,
                             "caption": caption_for(r["domain"], attr, value), "label": label})
    coco = [r for r in records if r["domain"] == "coco" and "captions" in r]
    for r in coco:
        for _ in range(50):
            other = rng.choice(coco)
            if other is not r and not (other["cats"] & r["cats"]):
                rows.append({"image": r["image"], "group": r["image"], "domain": "coco", "attribute": "scene",
                             "caption": rng.choice(r["captions"]), "label": 1})
                rows.append({"image": r["image"], "group": r["image"], "domain": "coco", "attribute": "scene",
                             "caption": rng.choice(other["captions"]), "label": 0})
                break
    return rows


def split_by_image(rows, rng):
    by_domain = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row["group"])
    split_of = {}
    for domain, imgs in by_domain.items():
        unique = sorted(set(imgs))
        rng.shuffle(unique)
        n_train, n_val = int(0.70 * len(unique)), int(0.15 * len(unique))
        for i, img in enumerate(unique):
            split_of[img] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return split_of


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-domain", type=int, default=3000)
    parser.add_argument("--skip-coco", action="store_true")
    parser.add_argument("--no-person-filter", action="store_true")
    args = parser.parse_args()
    rng = random.Random(SEED)

    global PEOPLE
    if not args.no_person_filter:
        from person_filter import PersonFilter
        print("Loading person detector (the first run downloads about 75 MB)...")
        PEOPLE = PersonFilter(OUT / "person_filter_cache.json")

    if not DATA.exists():
        raise SystemExit(f"Can't find {DATA}. Put this script in the FYP folder, next to Data/.")

    print(f"FYP folder: {ROOT}\n\nLoading datasets (real labels only)...")
    records = []
    records += load_objects(args.per_domain, rng)
    records += load_cars(args.per_domain, rng)
    records += load_dogs(args.per_domain, rng)
    records += load_cats(args.per_domain, rng)
    records += load_flowers(args.per_domain, rng)
    if not args.skip_coco:
        records += load_coco(args.per_domain, rng)

    if PEOPLE is not None:
        PEOPLE.save()
        removed = sorted(set(PEOPLE.removed))
        (OUT / "removed_people.txt").write_text("\n".join(removed) + "\n")
        print(f"\nPerson filter: removed {len(removed)} photos with people "
              f"(list saved in processed_data/removed_people.txt)")

    rows = make_pairs(records, rng)
    if not rows:
        raise SystemExit("\nNo pairs created. Check the messages above.")

    split_of = split_by_image(rows, rng)
    OUT.mkdir(exist_ok=True)
    fields = ["image_path", "caption", "label", "domain", "attribute"]
    writers, handles = {}, []
    for split in ("train", "val", "test"):
        fh = open(OUT / f"{split}.csv", "w", newline="")
        handles.append(fh)
        writers[split] = csv.DictWriter(fh, fieldnames=fields)
        writers[split].writeheader()

    counts = Counter()
    rng.shuffle(rows)
    for row in rows:
        split = split_of[row["group"]]
        writers[split].writerow({
            "image_path": row["image"].relative_to(DATA).as_posix(),
            "caption": row["caption"], "label": row["label"],
            "domain": row["domain"], "attribute": row["attribute"],
        })
        counts[(split, row["domain"], row["label"])] += 1
    for fh in handles:
        fh.close()

    print("\n" + "=" * 62)
    print(f"{'domain':10}{'train':>10}{'val':>10}{'test':>10}   consistent/inconsistent")
    for d in sorted({d for _, d, _ in counts}):
        per = [sum(counts[(s, d, l)] for l in (0, 1)) for s in ("train", "val", "test")]
        pos = sum(counts[(s, d, 1)] for s in ("train", "val", "test"))
        neg = sum(counts[(s, d, 0)] for s in ("train", "val", "test"))
        print(f"{d:10}{per[0]:>10}{per[1]:>10}{per[2]:>10}   {pos}/{neg}")
    totals = [sum(v for (s, _, _), v in counts.items() if s == sp) for sp in ("train", "val", "test")]
    print(f"{'TOTAL':10}{totals[0]:>10}{totals[1]:>10}{totals[2]:>10}")
    print("=" * 62)

    print("\nExample pairs (open the image and check the caption is right):")
    shown = set()
    for row in rows:
        if row["domain"] not in shown and row["label"] == 1:
            shown.add(row["domain"])
            print(f"  [{row['domain']}] {row['image'].relative_to(DATA)}\n      \"{row['caption']}\"")
    print(f"\nSaved train.csv, val.csv, test.csv in {OUT}")


if __name__ == "__main__":
    main()