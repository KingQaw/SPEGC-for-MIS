#!/usr/bin/env python3
"""
Generate a tiny synthetic "fundus-like" COCO instance-segmentation dataset.

The SPEGC release ships no image data at all (only annotation JSONs whose image
files are absent), so this script fabricates a small, deterministic,
self-contained dataset whose *schema* matches what the code expects.  It exists
purely to verify that the pipeline runs end to end -- it is NOT a benchmark and
metrics computed on it are meaningless.

Output layout (matches the registration in data/datasets/builtin.py)::

    datasets/synthetic/images/<name>.jpg
    datasets/synthetic/train.json
    datasets/synthetic/test.json

Usage::

    python tools/make_synthetic_fundus.py                # 8 train / 4 test
    python tools/make_synthetic_fundus.py --num-train 4 --num-test 2
"""

import argparse
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw

CLASS_NAME = "lesion"
HEIGHT, WIDTH = 320, 320
POLYGON_POINTS = 24  # -> 48 coordinates, comfortably >= the 6 the loader requires


def ellipse_polygon(cx, cy, rx, ry, n=POLYGON_POINTS, rotation=0.0):
    """Return a flattened [x0, y0, x1, y1, ...] polygon approximating an ellipse."""
    angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    cos_r, sin_r = math.cos(rotation), math.sin(rotation)
    xs = cx + rx * np.cos(angles)
    ys = cy + ry * np.sin(angles)
    # rotate about the centre
    xs_r = cx + (xs - cx) * cos_r - (ys - cy) * sin_r
    ys_r = cy + (xs - cx) * sin_r + (ys - cy) * cos_r
    return [round(float(v), 2) for pair in zip(xs_r, ys_r) for v in pair]


def polygon_bbox(poly):
    xs, ys = poly[0::2], poly[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return [x0, y0, x1 - x0, y1 - y0]


def polygon_area(poly):
    xs, ys = poly[0::2], poly[1::2]
    n = len(xs)
    acc = 0.0
    for i in range(n):
        j = (i + 1) % n
        acc += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(acc) / 2.0


def draw_fundus(rng):
    """Render one synthetic retinal fundus image plus its lesion polygons."""
    img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = WIDTH / 2.0, HEIGHT / 2.0
    radius = 0.46 * min(WIDTH, HEIGHT)
    tone = tuple(int(v) for v in rng.integers(110, 150, size=3))
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=tone)

    # a few "blood vessels" radiating from the optic disc
    disc_x = cx + rng.uniform(-0.22, 0.22) * WIDTH
    disc_y = cy + rng.uniform(-0.22, 0.22) * HEIGHT
    for _ in range(6):
        angle = rng.uniform(0, 2 * math.pi)
        r0 = 0.05 * radius
        r1 = rng.uniform(0.5, 1.0) * radius
        draw.line(
            [
                disc_x + r0 * math.cos(angle),
                disc_y + r0 * math.sin(angle),
                disc_x + r1 * math.cos(angle),
                disc_y + r1 * math.sin(angle),
            ],
            fill=(90, 40, 40),
            width=3,
        )
    draw.ellipse(
        [disc_x - 0.09 * radius, disc_y - 0.09 * radius,
         disc_x + 0.09 * radius, disc_y + 0.09 * radius],
        fill=(225, 210, 170),
    )

    # foreground: 1-3 dark lesions, well inside the retinal disc
    polygons = []
    for _ in range(int(rng.integers(1, 4))):
        ang = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(0.15, 0.6) * radius
        lx = cx + dist * math.cos(ang)
        ly = cy + dist * math.sin(ang)
        rx = rng.uniform(0.07, 0.15) * radius
        ry = rng.uniform(0.07, 0.15) * radius
        rot = rng.uniform(0, math.pi)
        poly = ellipse_polygon(lx, ly, rx, ry, rotation=rot)
        polygons.append(poly)
        draw.polygon(
            list(zip(poly[0::2], poly[1::2])),
            fill=(int(rng.integers(60, 100)), 15, 15),
        )

    # mild sensor noise
    arr = np.asarray(img).astype(np.int16)
    arr += rng.integers(-8, 9, size=arr.shape, dtype=np.int16)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), polygons


def build_split(out_dir, split, count, seed):
    rng = np.random.default_rng(seed)
    images, annotations = [], []
    ann_id = 1
    for i in range(count):
        img, polygons = draw_fundus(rng)
        file_name = "{}_{:03d}.jpg".format(split, i)
        img.save(os.path.join(out_dir, "images", file_name), quality=92)
        image_id = i + 1
        images.append(
            {"file_name": file_name, "height": HEIGHT, "width": WIDTH, "id": image_id}
        )
        for poly in polygons:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "iscrowd": 0,
                    "ignore": 0,
                    "bbox": polygon_bbox(poly),
                    "area": polygon_area(poly),
                    "segmentation": [poly],
                }
            )
            ann_id += 1
    data = {
        "type": "instance",
        "categories": [{"id": 1, "name": CLASS_NAME}],
        "images": images,
        "annotations": annotations,
    }
    path = os.path.join(out_dir, "{}.json".format(split))
    with open(path, "w") as fh:
        json.dump(data, fh, indent=1)
    return len(images), len(annotations), path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-train", type=int, default=8)
    parser.add_argument("--num-test", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260323)
    parser.add_argument(
        "--out-dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "datasets",
            "synthetic",
        ),
    )
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)
    for split, count, seed in (
        ("train", args.num_train, args.seed),
        ("test", args.num_test, args.seed + 1),
    ):
        n_img, n_ann, path = build_split(args.out_dir, split, count, seed)
        print(
            "[synth] {:<5} {:>3} images / {:>3} annotations -> {}".format(
                split, n_img, n_ann, path
            )
        )


if __name__ == "__main__":
    main()
