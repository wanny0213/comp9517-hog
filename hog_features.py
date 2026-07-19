"""
HOG feature extraction.

Descriptor: pixels_per_cell=(16,16), cells_per_block=(2,2), orientations=9
Image size:  128x128  →  vector length = 7*7*2*2*9 = 1764
"""
import numpy as np
from PIL import Image
from skimage.feature import hog


IMAGE_SIZE = (128, 128)
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    channel_axis=-1,
)
FEATURE_DIM = 1764


def extract_hog(image_array: np.ndarray) -> np.ndarray:
    """Extract HOG from a single (H, W, 3) float32 image in [0,1].

    Returns a 1-D float32 vector of length FEATURE_DIM.
    """
    return hog(image_array, **HOG_PARAMS).astype(np.float32)


def extract_hog_from_path(path: str) -> np.ndarray:
    """Load an image from disk, resize, and extract HOG."""
    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return extract_hog(arr)


def extract_hog_batch(paths: list[str], verbose: bool = True) -> np.ndarray:
    """Extract HOG from a list of paths. Returns (N, FEATURE_DIM) array."""
    features = []
    for i, p in enumerate(paths):
        if verbose and i % 500 == 0:
            print(f"  HOG: {i}/{len(paths)}", flush=True)
        try:
            features.append(extract_hog_from_path(p))
        except Exception as e:
            print(f"  [warn] {p}: {e} — using zeros")
            features.append(np.zeros(FEATURE_DIM, dtype=np.float32))
    return np.stack(features)


def extract_hog_from_array_batch(images: np.ndarray, verbose: bool = True) -> np.ndarray:
    """Extract HOG from a (N, H, W, 3) array. Returns (N, FEATURE_DIM)."""
    features = []
    for i, img in enumerate(images):
        if verbose and i % 500 == 0:
            print(f"  HOG: {i}/{len(images)}", flush=True)
        features.append(extract_hog(img))
    return np.stack(features)
