"""
Robustness evaluation for HOG + SVM.

Applies each degradation × severity to the test set (no retraining),
measures top-1 accuracy and macro-F1, and outputs a summary table + plots.

Usage:
    python evaluate_robustness.py \
        --model  results/hog/hog_svm.joblib \
        --test   dataset/test \
        --classes results/hog/class_names.json \
        --out_dir results/hog/robustness
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support

from degradations import DEGRADATION_TYPES, SEVERITY_LEVELS, apply_degradation
from hog_features import extract_hog, IMAGE_SIZE
from utils import load_image_paths


def load_test_images(root: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load test images as float32 arrays without extracting HOG yet."""
    paths, labels, class_names = load_image_paths(root)
    images = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize(IMAGE_SIZE)
        images.append(np.array(img, dtype=np.float32) / 255.0)
    return np.stack(images), labels, class_names


def evaluate_on_degraded(
    model,
    images: np.ndarray,
    labels: np.ndarray,
    degradation: str,
    severity: int,
) -> dict:
    degraded = np.stack([apply_degradation(img, degradation, severity) for img in images])
    features = np.stack([extract_hog(img) for img in degraded])

    t0 = time.time()
    scores = model.decision_function(features) if hasattr(model, "decision_function") else model.predict_proba(features)
    preds = np.argmax(scores, axis=1)
    infer_time = time.time() - t0

    top1 = float(np.mean(preds == labels))
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"top1_acc": top1, "macro_f1": float(f1), "infer_time_s": infer_time}


def plot_robustness(df: pd.DataFrame, metric: str, out_path: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    for deg in DEGRADATION_TYPES:
        subset = df[df["degradation"] == deg].sort_values("severity")
        ax.plot(subset["severity"], subset[metric], marker="o", label=deg)

    baseline = df[df["degradation"] == "clean"][metric].values
    if len(baseline):
        ax.axhline(baseline[0], color="black", linestyle="--", label="clean (baseline)")

    ax.set_xlabel("Severity Level")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"HOG+SVM Robustness — {metric}")
    ax.legend(fontsize=8)
    ax.set_xticks(SEVERITY_LEVELS)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to hog_svm.joblib")
    parser.add_argument("--test", required=True, help="Path to test split folder")
    parser.add_argument("--classes", required=True, help="Path to class_names.json")
    parser.add_argument("--out_dir", default="results/hog/robustness")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model …")
    model = joblib.load(args.model)

    print("Loading test images …")
    images, labels, _ = load_test_images(args.test)
    print(f"  {len(images)} test images")

    records = []

    # baseline (clean images)
    print("\nEvaluating on clean images …")
    features_clean = np.stack([extract_hog(img) for img in images])
    scores_clean = model.decision_function(features_clean) if hasattr(model, "decision_function") else model.predict_proba(features_clean)
    preds_clean = np.argmax(scores_clean, axis=1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds_clean, average="macro", zero_division=0)
    records.append({
        "degradation": "clean",
        "severity": 0,
        "top1_acc": float(np.mean(preds_clean == labels)),
        "macro_f1": float(f1),
        "infer_time_s": 0.0,
    })
    print(f"  Clean  top-1: {records[-1]['top1_acc']:.4f}  macro-F1: {records[-1]['macro_f1']:.4f}")

    for deg in DEGRADATION_TYPES:
        print(f"\nDegradation: {deg}")
        for sev in SEVERITY_LEVELS:
            metrics = evaluate_on_degraded(model, images, labels, deg, sev)
            records.append({"degradation": deg, "severity": sev, **metrics})
            print(f"  sev={sev}  top-1={metrics['top1_acc']:.4f}  macro-F1={metrics['macro_f1']:.4f}")

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "robustness_results.csv", index=False)

    # pretty table to stdout
    print("\n" + "=" * 60)
    print(df.pivot_table(index="degradation", columns="severity", values="top1_acc").to_string())

    plot_robustness(df, "top1_acc", str(out_dir / "robustness_top1.png"))
    plot_robustness(df, "macro_f1", str(out_dir / "robustness_macro_f1.png"))

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
