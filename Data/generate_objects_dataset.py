"""
Generate the synthetic objects dataset for the project.
"""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUTPUT_DIR = Path(__file__).resolve().parent / "synthetic_objects"
IMAGES_DIR = OUTPUT_DIR / "images"
NUM_IMAGES = 2000
SEED = 42

SHAPES = ["circle", "square", "triangle", "rectangle", "star"]
COLOURS = {
    "red": (220, 30, 30),
    "blue": (30, 70, 220),
    "green": (30, 160, 60),
    "yellow": (240, 210, 20),
    "orange": (245, 140, 20),
    "purple": (130, 40, 160),
}
POSITIONS = {"left": 64, "centre": 128, "right": 192}


def draw_shape(draw, shape, x, y, size, rgb):
    if shape == "circle":
        draw.ellipse([x - size, y - size, x + size, y + size], fill=rgb)
    elif shape == "square":
        draw.rectangle([x - size, y - size, x + size, y + size], fill=rgb)
    elif shape == "triangle":
        draw.polygon([(x, y - size), (x - size, y + size), (x + size, y + size)], fill=rgb)
    elif shape == "rectangle":
        draw.rectangle([x - size, y - size // 2, x + size, y + size // 2], fill=rgb)
    elif shape == "star":
        points = []
        for i in range(10):
            angle = i * np.pi / 5
            r = size if i % 2 == 0 else size / 2
            points.append((x + r * np.sin(angle), y - r * np.cos(angle)))
        draw.polygon(points, fill=rgb)


def main():
    rng = random.Random(SEED)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating {NUM_IMAGES} images into {OUTPUT_DIR}")

    annotations = []
    for i in range(NUM_IMAGES):
        shape = rng.choice(SHAPES)
        colour = rng.choice(list(COLOURS))
        position = rng.choice(list(POSITIONS))

        shade = rng.randint(225, 255)
        img = Image.new("RGB", (256, 256), (shade, shade, shade))
        size = rng.randint(28, 50)
        x = POSITIONS[position] + rng.randint(-8, 8)
        y = 128 + rng.randint(-50, 50)
        draw_shape(ImageDraw.Draw(img), shape, x, y, size, COLOURS[colour])

        name = f"object_{i:05d}.png"
        img.save(IMAGES_DIR / name)
        annotations.append({"image_id": i, "image_name": name, "object_type": shape,
                            "color": colour, "position": position})
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{NUM_IMAGES}")

    with open(OUTPUT_DIR / "annotations.json", "w") as f:
        json.dump(annotations, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()