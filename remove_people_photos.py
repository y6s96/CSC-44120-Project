"""
Move every photo that contains a person OUT of the dataset folders.
Photos are moved (not deleted) to FYP/removed_people_photos/, keeping their folder
structure, so nothing is lost if the detector makes a mistake.

Run from the FYP folder:
    python3 remove_people_photos.py              # checks CARS, DOGS, CATS and the flowers
    python3 remove_people_photos.py DOGS         # just one folder
You can stop it with Ctrl+C and run it again later: checked photos are remembered.
"""

import argparse
import re
import shutil
import time
from pathlib import Path

from PIL import Image

from person_filter import PersonFilter

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data"
MOVED_TO = ROOT / "removed_people_photos"
CACHE = ROOT / "processed_data" / "person_filter_cache.json"
IMG_EXT = {".jpg", ".jpeg", ".png"}


def readable(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def photos_in(name):
    if name == "FLOWERS":
        return sorted(p for p in DATA.rglob("image_*.jpg")
                      if "venv" not in p.parts and re.fullmatch(r"image_\d{5}\.jpg", p.name))
    folder = DATA / name
    photos = []
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() not in IMG_EXT or "venv" in p.parts:
            continue
        if p.name.startswith("COCO_"):
            continue
        if name == "CATS" and any(part.lower() in {"dog", "dogs"} for part in p.parts):
            continue
        photos.append(p)
    return photos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="*", default=["CARS", "DOGS", "CATS", "FLOWERS"])
    parser.add_argument("--gpu", action="store_true", help="use the Mac GPU (much faster)")
    parser.add_argument("--compare", type=int, default=0, help="check GPU vs CPU on N photos, move nothing")
    args = parser.parse_args()

    print("Loading person detector (the first run downloads about 75 MB)...")
    if args.compare:
        sample = [p for p in photos_in("DOGS")[::150] if readable(p)][:args.compare]
        cpu = PersonFilter(ROOT / "processed_data" / "_compare_cpu.json")
        gpu = PersonFilter(ROOT / "processed_data" / "_compare_gpu.json", device="mps")
        cpu.cache, gpu.cache = {}, {}
        gpu.has_person(sample[0])
        gpu.cache = {}
        t = time.time(); a = [cpu.has_person(p) for p in sample]; t_cpu = time.time() - t
        t = time.time(); b = [gpu.has_person(p) for p in sample]; t_gpu = time.time() - t
        same = sum(x == y for x, y in zip(a, b))
        print(f"\nChecked {len(sample)} dog photos ({sum(a)} with people according to CPU)")
        print(f"GPU and CPU agree on {same}/{len(sample)} photos")
        print(f"CPU {t_cpu / len(sample):.2f}s per photo, GPU {t_gpu / len(sample):.2f}s per photo")
        print("Nothing was moved.")
        return

    pf = PersonFilter(CACHE, device="mps" if args.gpu else "cpu")
    print("Using the Mac GPU" if args.gpu else "Using the CPU")
    log_path = MOVED_TO / "moved_photos.txt"
    MOVED_TO.mkdir(exist_ok=True)

    total = 0
    for name in args.folders:
        if name != "FLOWERS" and not (DATA / name).exists():
            print(f"\n{name}: Data/{name} not found, skipping")
            continue
        photos = photos_in(name)
        todo = sum(1 for p in photos if p.as_posix() not in pf.cache)
        print(f"\n{name}: {len(photos)} photos, {todo} still to check")

        moved, start, checked_now = 0, time.time(), 0
        try:
            with open(log_path, "a") as log:
                for i, photo in enumerate(photos, 1):
                    new = photo.as_posix() not in pf.cache
                    if not readable(photo):
                        continue
                    if pf.has_person(photo):
                        dest = MOVED_TO / photo.relative_to(DATA)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(photo), str(dest))
                        log.write(photo.relative_to(DATA).as_posix() + "\n")
                        moved += 1
                    checked_now += new
                    if i % 500 == 0:
                        left = todo - checked_now
                        rate = checked_now / max(time.time() - start, 1e-9)
                        eta = f", about {left / rate / 60:.0f} min left" if checked_now and left else ""
                        print(f"   {i}/{len(photos)} done, {moved} moved{eta}")
        finally:
            pf.save()
        print(f"{name}: moved {moved} photos with people to removed_people_photos/")
        total += moved

    print(f"\nDone. {total} photos with people moved out of Data/.")
    print(f"Full list: {log_path}")
    print("Now rebuild the training data:  python3 process_all_datasets.py --skip-coco")


if __name__ == "__main__":
    main()
