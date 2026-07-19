"""
Train SIFT + BoVW + SVM classifier (Alex's pipeline).

Usage:
    python train_classifier.py --train dataset/train --test dataset/test

Outputs saved to --out_dir:
    bovw_vocab.npy        vocabulary cluster centres
    bovw_svm.joblib       trained SVM
    class_names.json      species list
    results.json          metrics
    confusion_matrix.png
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.svm import SVC

from features import build_vocabulary, extract_bovw_batch, DEFAULT_VOCAB_SIZE
from utils import load_image_paths


def top_k_accuracy(proba: np.ndarray, labels: np.ndarray, k: int) -> float:
    top_k = np.argsort(proba, axis=1)[:, -k:]
    return np.mean([labels[i] in top_k[i] for i in range(len(labels))])


def plot_confusion_matrix(cm, class_names, out_path, max_classes=40):
    n = min(len(class_names), max_classes)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.4), max(7, n * 0.4)))
    im = ax.imshow(cm[:n, :n], interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    if n <= 40:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names[:n], rotation=90, fontsize=6)
        ax.set_yticklabels(class_names[:n], fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — SIFT+BoVW+SVM (first {n} classes)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", default=None)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out_dir", default="results/bovw")
    parser.add_argument("--vocab_size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--svm_c", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading train paths …")
    train_paths, y_train, class_names = load_image_paths(args.train)

    vocab = build_vocabulary(train_paths, vocab_size=args.vocab_size)
    np.save(out_dir / "bovw_vocab.npy", vocab)
    print(f"Vocabulary saved ({len(vocab)} words).")

    print("Encoding train images …")
    X_train = extract_bovw_batch(train_paths, vocab)

    print("Loading test paths …")
    test_paths, y_test, _ = load_image_paths(args.test)
    print("Encoding test images …")
    X_test = extract_bovw_batch(test_paths, vocab)

    with open(out_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    print("\nTraining SVM …")
    t0 = time.time()
    svm = SVC(C=args.svm_c, kernel="rbf", probability=True, random_state=42)
    svm.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Train time: {train_time:.1f}s")

    t0 = time.time()
    test_proba = svm.predict_proba(X_test)
    infer_time = time.time() - t0
    preds = np.argmax(test_proba, axis=1)

    p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, preds)

    results = {
        "top1_acc": float(np.mean(preds == y_test)),
        "top5_acc": float(top_k_accuracy(test_proba, y_test, 5)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "train_time_s": train_time,
        "infer_time_s": infer_time,
    }
    print(f"  Test top-1: {results['top1_acc']:.4f}  top-5: {results['top5_acc']:.4f}  macro-F1: {results['macro_f1']:.4f}")
    print(classification_report(y_test, preds, target_names=class_names, zero_division=0))

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_confusion_matrix(cm, class_names, str(out_dir / "confusion_matrix.png"))
    joblib.dump(svm, out_dir / "bovw_svm.joblib")
    print(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    main()
