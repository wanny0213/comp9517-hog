"""
HOG descriptor ablation study.

Systematically varies pixels_per_cell and orientations one at a time,
measuring val-set top-1 accuracy and macro-F1 for each configuration.

This is the ablation required for comprehensive marks (23-26).
Keeps all other variables (SVM C, image size, cells_per_block) fixed.

Usage:
    python ablation_hog.py --train dataset/train --val dataset/val --out_dir results/ablation
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import LinearSVC

from utils import load_image_paths

try:
    from skimage.feature import hog
    from PIL import Image
except ImportError as e:
    raise SystemExit(f"Missing dependency: {e}")


IMAGE_SIZE = (128, 128)
BASE_CELLS_PER_BLOCK = (2, 2)
BASE_BLOCK_NORM = "L2-Hys"

# ── sweep ranges ──────────────────────────────────────────────────────────────
PIXELS_PER_CELL_OPTIONS = [(8, 8), (12, 12), (16, 16), (24, 24), (32, 32)]
ORIENTATIONS_OPTIONS    = [6, 8, 9, 12, 16]

# Base config (used as the "fixed" axis while the other varies)
BASE_PPC  = (16, 16)
BASE_ORI  = 9


def extract_hog_custom(paths: list[str], ppc: tuple, ori: int) -> np.ndarray:
    features = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize(IMAGE_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
        feat = hog(
            arr,
            orientations=ori,
            pixels_per_cell=ppc,
            cells_per_block=BASE_CELLS_PER_BLOCK,
            block_norm=BASE_BLOCK_NORM,
            channel_axis=-1,
        ).astype(np.float32)
        features.append(feat)
    return np.stack(features)


def evaluate(X_train, y_train, X_val, y_val) -> dict:
    clf = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    preds = clf.predict(X_val)
    p, r, f1, _ = precision_recall_fscore_support(y_val, preds, average="macro", zero_division=0)
    return {
        "top1_acc": float(np.mean(preds == y_val)),
        "macro_f1": float(f1),
        "train_time_s": train_time,
        "feature_dim": int(X_train.shape[1]),
    }


def run_sweep(train_paths, y_train, val_paths, y_val, param_name: str,
              options: list, fixed_ppc: tuple, fixed_ori: int, desc: str) -> list[dict]:
    results = []
    for opt in options:
        ppc = opt if param_name == "pixels_per_cell" else fixed_ppc
        ori = opt if param_name == "orientations"    else fixed_ori
        label = str(opt)
        print(f"  {desc}={label}  ppc={ppc}  ori={ori} …", flush=True)

        X_tr = extract_hog_custom(train_paths, ppc, ori)
        X_va = extract_hog_custom(val_paths,   ppc, ori)
        metrics = evaluate(X_tr, y_train, X_va, y_val)
        results.append({"param": label, "ppc": str(ppc), "ori": ori, **metrics})
        print(f"    top-1={metrics['top1_acc']:.4f}  macro-F1={metrics['macro_f1']:.4f}  "
              f"dim={metrics['feature_dim']}  train={metrics['train_time_s']:.1f}s")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",   required=True, help="Path to train split")
    parser.add_argument("--val",     required=True, help="Path to val split")
    parser.add_argument("--out_dir", default="results/ablation")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading image paths …")
    train_paths, y_train, class_names = load_image_paths(args.train)
    val_paths,   y_val,   _           = load_image_paths(args.val)
    print(f"  {len(train_paths)} train, {len(val_paths)} val, {len(class_names)} classes")

    all_results = {}

    # ── sweep 1: pixels_per_cell (orientations fixed at base) ────────────────
    print("\n[Sweep 1] Varying pixels_per_cell (orientations fixed at 9)")
    all_results["pixels_per_cell"] = run_sweep(
        train_paths, y_train, val_paths, y_val,
        param_name="pixels_per_cell",
        options=PIXELS_PER_CELL_OPTIONS,
        fixed_ppc=BASE_PPC,
        fixed_ori=BASE_ORI,
        desc="ppc",
    )

    # ── sweep 2: orientations (pixels_per_cell fixed at base) ─────────────────
    print("\n[Sweep 2] Varying orientations (pixels_per_cell fixed at (16,16))")
    all_results["orientations"] = run_sweep(
        train_paths, y_train, val_paths, y_val,
        param_name="orientations",
        options=ORIENTATIONS_OPTIONS,
        fixed_ppc=BASE_PPC,
        fixed_ori=BASE_ORI,
        desc="ori",
    )

    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # pretty summary tables
    print("\n=== pixels_per_cell sweep ===")
    print(f"{'ppc':<14} {'top-1':>8} {'macro-F1':>10} {'dim':>8} {'train(s)':>10}")
    for r in all_results["pixels_per_cell"]:
        print(f"{r['ppc']:<14} {r['top1_acc']:>8.4f} {r['macro_f1']:>10.4f} "
              f"{r['feature_dim']:>8} {r['train_time_s']:>10.1f}")

    print("\n=== orientations sweep ===")
    print(f"{'ori':<8} {'top-1':>8} {'macro-F1':>10} {'dim':>8} {'train(s)':>10}")
    for r in all_results["orientations"]:
        print(f"{r['ori']:<8} {r['top1_acc']:>8.4f} {r['macro_f1']:>10.4f} "
              f"{r['feature_dim']:>8} {r['train_time_s']:>10.1f}")

    print(f"\nResults saved to {out_dir}/ablation_results.json")


if __name__ == "__main__":
    main()
