"""
Shared utilities: load a dataset from a species_name/img.jpg folder structure.
Expected layout:
    root/
        Homo_sapiens/
            0001.jpg
            0002.jpg
        Canis_lupus/
            ...
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image


def load_dataset(root: str, image_size: tuple[int, int] = (128, 128)):
    """Return (images, labels, class_names) from a species folder tree.

    images     — float32 array (N, H, W, 3), normalised to [0, 1]
    labels     — int32 array (N,)
    class_names — list of species name strings, index = class id
    """
    root = Path(root)
    class_names = sorted(
        d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not class_names:
        raise ValueError(f"No subdirectories found in {root}")

    images, labels = [], []
    for cls_id, cls_name in enumerate(class_names):
        cls_dir = root / cls_name
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                img = Image.open(img_path).convert("RGB").resize(image_size)
                images.append(np.array(img, dtype=np.float32) / 255.0)
                labels.append(cls_id)
            except Exception as e:
                print(f"  [warn] skipping {img_path}: {e}")

    return np.stack(images), np.array(labels, dtype=np.int32), class_names


def load_image_paths(root: str):
    """Return (paths, labels, class_names) without loading pixel data.

    Useful when feature extractors want to open images themselves.
    """
    root = Path(root)
    class_names = sorted(
        d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    paths, labels = [], []
    for cls_id, cls_name in enumerate(class_names):
        cls_dir = root / cls_name
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            paths.append(str(img_path))
            labels.append(cls_id)

    return paths, np.array(labels, dtype=np.int32), class_names
