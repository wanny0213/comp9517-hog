"""
Synthetic dataset generator — for testing the pipeline without real data.

Creates a tiny dataset of coloured-shape images so you can verify the full
HOG+SVM pipeline runs end-to-end in seconds.

Usage:
    python demo.py                        # creates dataset/ then trains+evaluates
    python demo.py --only_data            # just generate images
    python demo.py --n_classes 20 --n_train 40 --n_test 10
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SHAPE_NAMES = [
    "circle", "square", "triangle", "ellipse", "rectangle",
    "hexagon", "star", "cross", "diamond", "trapezoid",
]


def random_colour(rng: np.random.Generator) -> tuple:
    return tuple(rng.integers(40, 220, size=3).tolist())


def draw_shape(name: str, size: int, rng: np.random.Generator) -> Image.Image:
    img = Image.new("RGB", (size, size), color=tuple(rng.integers(180, 255, 3).tolist()))
    draw = ImageDraw.Draw(img)
    m = size // 6
    colour = random_colour(rng)

    if name == "circle":
        draw.ellipse([m, m, size - m, size - m], fill=colour)
    elif name == "square":
        draw.rectangle([m, m, size - m, size - m], fill=colour)
    elif name == "triangle":
        draw.polygon([(size // 2, m), (m, size - m), (size - m, size - m)], fill=colour)
    elif name == "ellipse":
        draw.ellipse([m, size // 3, size - m, size - size // 3], fill=colour)
    elif name == "rectangle":
        draw.rectangle([m, size // 3, size - m, size - size // 3], fill=colour)
    elif name == "hexagon":
        pts = [(size // 2 + int((size // 2 - m) * np.cos(np.pi / 3 * i)),
                size // 2 + int((size // 2 - m) * np.sin(np.pi / 3 * i))) for i in range(6)]
        draw.polygon(pts, fill=colour)
    elif name == "star":
        pts = []
        for i in range(10):
            r = (size // 2 - m) if i % 2 == 0 else (size // 4 - m // 2)
            angle = np.pi / 5 * i - np.pi / 2
            pts.append((size // 2 + int(r * np.cos(angle)),
                         size // 2 + int(r * np.sin(angle))))
        draw.polygon(pts, fill=colour)
    elif name == "cross":
        w = size // 4
        draw.rectangle([size // 2 - w // 2, m, size // 2 + w // 2, size - m], fill=colour)
        draw.rectangle([m, size // 2 - w // 2, size - m, size // 2 + w // 2], fill=colour)
    elif name == "diamond":
        draw.polygon([(size // 2, m), (size - m, size // 2),
                      (size // 2, size - m), (m, size // 2)], fill=colour)
    elif name == "trapezoid":
        draw.polygon([(size // 4, m), (size - size // 4, m),
                      (size - m, size - m), (m, size - m)], fill=colour)
    else:
        draw.ellipse([m, m, size - m, size - m], fill=colour)

    return img


def generate_dataset(root: Path, class_names: list[str], n_train: int, n_val: int,
                     n_test: int, img_size: int = 128, seed: int = 0):
    rng = np.random.default_rng(seed)
    for split, count in [("train", n_train), ("val", n_val), ("test", n_test)]:
        for cls in class_names:
            split_dir = root / split / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                img = draw_shape(cls, img_size, rng)
                img.save(split_dir / f"{i:04d}.jpg")
    print(f"Synthetic dataset written to {root}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--n_train", type=int, default=40)
    parser.add_argument("--n_val", type=int, default=10)
    parser.add_argument("--n_test", type=int, default=10)
    parser.add_argument("--out", default="demo_dataset", help="Dataset root")
    parser.add_argument("--only_data", action="store_true")
    args = parser.parse_args()

    class_names = SHAPE_NAMES[: args.n_classes]
    root = Path(args.out)
    if root.exists():
        shutil.rmtree(root)

    generate_dataset(root, class_names, args.n_train, args.n_val, args.n_test)

    if args.only_data:
        return

    print("\nRunning HOG+SVM pipeline on synthetic data …")
    cmd = [
        sys.executable, "train_hog_classifier.py",
        "--train", str(root / "train"),
        "--val",   str(root / "val"),
        "--test",  str(root / "test"),
        "--out_dir", "demo_results",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
