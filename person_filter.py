"""Remove photos that contain people (data cleaning only).
Uses torchvision's ready-made Faster R-CNN person detector, pretrained on COCO.
It is ONLY used to clean the dataset before training. It is not part of the
consistency model, which is still trained from scratch."""

import json
from pathlib import Path

import torch
from PIL import Image

try:
    from torchvision.models.detection import (FasterRCNN_MobileNet_V3_Large_FPN_Weights,
                                              fasterrcnn_mobilenet_v3_large_fpn)
    from torchvision.transforms.functional import to_tensor
except ImportError:
    raise SystemExit("The person filter needs torchvision. Run:  pip3 install torchvision")


class PersonFilter:
    def __init__(self, cache_path, threshold=0.4, device="cpu"):
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        self.model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights, box_score_thresh=threshold, min_size=512, max_size=768).eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.person = weights.meta["categories"].index("person")
        self.cache_path = Path(cache_path)
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.removed = []
        self.new_checks = 0

    def has_person(self, path):
        key = Path(path).as_posix()
        if key not in self.cache:
            with Image.open(path) as im:
                im.draft("RGB", (768, 768))
                im = im.convert("RGB")
                im.thumbnail((768, 768))
                x = to_tensor(im)
            with torch.no_grad():
                out = self.model([x.to(self.device)])[0]
            self.cache[key] = bool((out["labels"].cpu() == self.person).any())
            self.new_checks += 1
            if self.new_checks % 100 == 0:
                print(f"      checked {self.new_checks} photos for people...")
                self.save()
        if self.cache[key]:
            self.removed.append(key)
        return self.cache[key]

    def save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache))
