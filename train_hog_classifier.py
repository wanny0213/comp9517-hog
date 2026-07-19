"""
Train HOG + LinearSVC (and Random Forest) on a species image dataset.

LinearSVC is used instead of RBF SVM because at 500+ classes × 40 imgs the
RBF kernel becomes O(n²·p) — several hours on Colab CPU. LinearSVC trains
in minutes at the same scale. Top-5 is derived from decision_function scores.

Usage:
    python train_hog_classifier.py --train dataset/train --val dataset/val --test dataset/test

Outputs (saved to --out_dir):
    hog_svm.joblib          trained LinearSVC
    hog_rf.joblib           trained Random Forest
    class_names.json        ordered list of species names
    results.json            all metrics
    svm_confusion_matrix.png
    rf_confusion_matrix.png
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.svm import LinearSVC

from hog_features import extract_hog_batch
from utils import load_image_paths


def top_k_accuracy_from_scores(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Works with both decision_function scores and predict_proba outputs."""
    top_k = np.argsort(scores, axis=1)[:, -k:]
    return float(np.mean([labels[i] in top_k[i] for i in range(len(labels))]))


def most_confused_pairs(cm: np.ndarray, class_names: list[str], n: int = 10):
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), class_names[i], class_names[j]))
    pairs.sort(reverse=True)
    return pairs[:n]


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: str, max_classes: int = 50):
    n = min(len(class_names), max_classes)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.35), max(7, n * 0.35)))
    im = ax.imshow(cm[:n, :n], interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    if n <= 50:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names[:n], rotation=90, fontsize=5)
        ax.set_yticklabels(class_names[:n], fontsize=5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (first {n} classes)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def load_split(root: str, desc: str):
    print(f"\nLoading {desc} images from {root} …")
    paths, labels, class_names = load_image_paths(root)
    print(f"  {len(paths)} images, {len(class_names)} classes")
    features = extract_hog_batch(paths, verbose=True)
    return features, labels, class_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",   required=True, help="Path to train split folder")
    parser.add_argument("--val",     default=None,  help="Path to val split folder (optional)")
    parser.add_argument("--test",    required=True, help="Path to test split folder")
    parser.add_argument("--out_dir", default="results/hog")
    parser.add_argument("--svm_c",   type=float, default=0.1,
                        help="LinearSVC regularisation C (0.1 is a good default for HOG)")
    parser.add_argument("--no_rf",   action="store_true", help="Skip Random Forest")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, class_names = load_split(args.train, "train")
    X_test,  y_test,  _           = load_split(args.test,  "test")
    if args.val:
        X_val, y_val, _ = load_split(args.val, "val")

    with open(out_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    results = {}

    # ── LinearSVC ─────────────────────────────────────────────────────────────
    print("\nTraining LinearSVC …")
    t0 = time.time()
    # CalibratedClassifierCV wraps LinearSVC to add predict_proba via cross-val
    # isotonic calibration — needed for proper probability estimates.
    # Use cv=3 to keep Colab runtime reasonable.
    base_svc = LinearSVC(C=args.svm_c, max_iter=2000, random_state=42)
    svm = CalibratedClassifierCV(base_svc, cv=3, method="isotonic")
    svm.fit(X_train, y_train)
    svm_train_time = time.time() - t0
    print(f"  Train time: {svm_train_time:.1f}s")

    def _eval(X, y, desc):
        t0 = time.time()
        scores = svm.predict_proba(X)
        infer_time = time.time() - t0
        preds = np.argmax(scores, axis=1)
        p, r, f1, _ = precision_recall_fscore_support(y, preds, average="macro", zero_division=0)
        return {
            "top1_acc":        float(np.mean(preds == y)),
            "top5_acc":        top_k_accuracy_from_scores(scores, y, 5),
            "macro_precision": float(p),
            "macro_recall":    float(r),
            "macro_f1":        float(f1),
            "infer_time_s":    infer_time,
        }, preds, scores

    if args.val:
        val_metrics, val_preds, _ = _eval(X_val, y_val, "val")
        results["svm_val"] = val_metrics
        print(f"  Val  top-1={val_metrics['top1_acc']:.4f}  "
              f"top-5={val_metrics['top5_acc']:.4f}  "
              f"macro-F1={val_metrics['macro_f1']:.4f}")

    test_metrics, test_preds, test_scores = _eval(X_test, y_test, "test")
    cm = confusion_matrix(y_test, test_preds)
    results["svm_test"] = {
        **test_metrics,
        "train_time_s": svm_train_time,
        "most_confused": most_confused_pairs(cm, class_names, n=10),
    }
    print(f"  Test top-1={test_metrics['top1_acc']:.4f}  "
          f"top-5={test_metrics['top5_acc']:.4f}  "
          f"macro-F1={test_metrics['macro_f1']:.4f}")
    print(classification_report(y_test, test_preds, target_names=class_names, zero_division=0))

    plot_confusion_matrix(cm, class_names, str(out_dir / "svm_confusion_matrix.png"))
    joblib.dump(svm, out_dir / "hog_svm.joblib")

    # ── Random Forest ─────────────────────────────────────────────────────────
    if not args.no_rf:
        print("\nTraining Random Forest …")
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train)
        rf_train_time = time.time() - t0
        print(f"  Train time: {rf_train_time:.1f}s")

        t0 = time.time()
        rf_scores = rf.predict_proba(X_test)
        rf_infer_time = time.time() - t0
        rf_preds = np.argmax(rf_scores, axis=1)
        p, r, f1, _ = precision_recall_fscore_support(y_test, rf_preds, average="macro", zero_division=0)
        rf_cm = confusion_matrix(y_test, rf_preds)

        results["rf_test"] = {
            "top1_acc":        float(np.mean(rf_preds == y_test)),
            "top5_acc":        top_k_accuracy_from_scores(rf_scores, y_test, 5),
            "macro_precision": float(p),
            "macro_recall":    float(r),
            "macro_f1":        float(f1),
            "train_time_s":    rf_train_time,
            "infer_time_s":    rf_infer_time,
            "most_confused":   most_confused_pairs(rf_cm, class_names, n=10),
        }
        print(f"  Test top-1={results['rf_test']['top1_acc']:.4f}  "
              f"macro-F1={results['rf_test']['macro_f1']:.4f}")
        plot_confusion_matrix(rf_cm, class_names, str(out_dir / "rf_confusion_matrix.png"))
        joblib.dump(rf, out_dir / "hog_rf.joblib")

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {out_dir}/")

    # ── summary printout ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"{'Method':<20} {'Top-1':>8} {'Top-5':>8} {'macro-F1':>10}")
    print("-" * 55)
    for key, label in [("svm_test", "LinearSVC"), ("rf_test", "Random Forest")]:
        if key in results:
            r = results[key]
            print(f"{label:<20} {r['top1_acc']:>8.4f} {r['top5_acc']:>8.4f} {r['macro_f1']:>10.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
