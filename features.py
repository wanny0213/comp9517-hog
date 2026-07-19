"""
SIFT + Bag-of-Visual-Words feature extraction (Alex's pipeline).

Workflow:
  1. extract_sift_descriptors()   — raw SIFT keypoint descriptors per image
  2. build_vocabulary()           — k-means over a sample of descriptors
  3. encode_bovw()                — histogram encoding per image
  4. extract_bovw_batch()         — convenience wrapper for a list of paths
"""
import numpy as np
import cv2
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans


IMAGE_SIZE = (128, 128)
DEFAULT_VOCAB_SIZE = 500


def _load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"Cannot read {path}")
    img = cv2.resize(img, IMAGE_SIZE)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def extract_sift_descriptors(path: str) -> np.ndarray | None:
    """Return (N, 128) SIFT descriptor array for one image, or None if no keypoints."""
    gray = _load_gray(path)
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(gray, None)
    return descriptors  # None if no keypoints found


def build_vocabulary(paths: list[str], vocab_size: int = DEFAULT_VOCAB_SIZE,
                     sample_per_image: int = 50, random_state: int = 42) -> np.ndarray:
    """Fit a k-means vocabulary on SIFT descriptors sampled from all training images.

    Returns cluster centres as (vocab_size, 128) float32 array.
    """
    print(f"Building BoVW vocabulary (k={vocab_size}) from {len(paths)} images …")
    all_desc = []
    rng = np.random.default_rng(random_state)
    for i, p in enumerate(paths):
        if i % 1000 == 0:
            print(f"  SIFT descriptors: {i}/{len(paths)}", flush=True)
        desc = extract_sift_descriptors(p)
        if desc is not None:
            if len(desc) > sample_per_image:
                idx = rng.choice(len(desc), sample_per_image, replace=False)
                desc = desc[idx]
            all_desc.append(desc)

    all_desc = np.vstack(all_desc).astype(np.float32)
    print(f"  Fitting k-means on {len(all_desc)} descriptors …")
    kmeans = MiniBatchKMeans(n_clusters=vocab_size, random_state=random_state,
                             batch_size=10_000, n_init=3)
    kmeans.fit(all_desc)
    return kmeans.cluster_centers_


def encode_bovw(path: str, vocab: np.ndarray) -> np.ndarray:
    """Encode one image as a normalised BoVW histogram."""
    vocab_size = len(vocab)
    desc = extract_sift_descriptors(path)
    if desc is None:
        return np.zeros(vocab_size, dtype=np.float32)
    dists = np.linalg.norm(desc[:, None, :] - vocab[None, :, :], axis=2)
    assignments = np.argmin(dists, axis=1)
    hist, _ = np.histogram(assignments, bins=vocab_size, range=(0, vocab_size))
    hist = hist.astype(np.float32)
    norm = np.linalg.norm(hist)
    return hist / norm if norm > 0 else hist


def extract_bovw_batch(paths: list[str], vocab: np.ndarray, verbose: bool = True) -> np.ndarray:
    """Encode a list of image paths. Returns (N, vocab_size) float32 array."""
    features = []
    for i, p in enumerate(paths):
        if verbose and i % 500 == 0:
            print(f"  BoVW encoding: {i}/{len(paths)}", flush=True)
        features.append(encode_bovw(p, vocab))
    return np.stack(features)
