"""
Test-time image degradations.

5 types × 5 severity levels.  NEVER applied during training.

All functions accept and return float32 numpy arrays in [0, 1] with shape (H, W, 3).
"""
import io

import cv2
import numpy as np
from PIL import Image


DEGRADATION_TYPES = [
    "gaussian_noise",
    "gaussian_blur",
    "motion_blur",
    "brightness_contrast",
    "jpeg_compression",
]

SEVERITY_LEVELS = [1, 2, 3, 4, 5]


# ── per-type severity parameters ────────────────────────────────────────────

_NOISE_STD = [0.02, 0.06, 0.12, 0.20, 0.35]
_BLUR_SIGMA = [0.5, 1.0, 2.0, 3.5, 5.0]
_MOTION_KERNEL = [3, 5, 9, 15, 21]
_BRIGHTNESS_DELTA = [0.1, 0.2, 0.35, 0.5, 0.7]   # additive brightness shift
_CONTRAST_FACTOR = [0.9, 0.75, 0.55, 0.35, 0.15]  # multiplier toward 0.5
_JPEG_QUALITY = [90, 70, 50, 30, 10]


def gaussian_noise(img: np.ndarray, severity: int) -> np.ndarray:
    std = _NOISE_STD[severity - 1]
    noisy = img + np.random.normal(0, std, img.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def gaussian_blur(img: np.ndarray, severity: int) -> np.ndarray:
    sigma = _BLUR_SIGMA[severity - 1]
    ksize = int(6 * sigma) | 1  # nearest odd number >= 6*sigma
    blurred = cv2.GaussianBlur((img * 255).astype(np.uint8), (ksize, ksize), sigma)
    return blurred.astype(np.float32) / 255.0


def motion_blur(img: np.ndarray, severity: int) -> np.ndarray:
    k = _MOTION_KERNEL[severity - 1]
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    blurred = cv2.filter2D((img * 255).astype(np.uint8), -1, kernel)
    return blurred.astype(np.float32) / 255.0


def brightness_contrast(img: np.ndarray, severity: int) -> np.ndarray:
    delta = _BRIGHTNESS_DELTA[severity - 1]
    factor = _CONTRAST_FACTOR[severity - 1]
    degraded = img * (1 - factor) + 0.5 * factor + delta
    return np.clip(degraded, 0.0, 1.0)


def jpeg_compression(img: np.ndarray, severity: int) -> np.ndarray:
    quality = _JPEG_QUALITY[severity - 1]
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    decoded = np.array(Image.open(buf), dtype=np.float32) / 255.0
    return decoded


# ── dispatch ─────────────────────────────────────────────────────────────────

_DISPATCH = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "brightness_contrast": brightness_contrast,
    "jpeg_compression": jpeg_compression,
}


def apply_degradation(img: np.ndarray, degradation: str, severity: int) -> np.ndarray:
    """Apply a named degradation at a given severity level (1–5)."""
    if degradation not in _DISPATCH:
        raise ValueError(f"Unknown degradation '{degradation}'. Choose from {DEGRADATION_TYPES}")
    if severity not in SEVERITY_LEVELS:
        raise ValueError(f"Severity must be in {SEVERITY_LEVELS}")
    return _DISPATCH[degradation](img, severity)


def apply_degradation_to_batch(
    images: np.ndarray, degradation: str, severity: int
) -> np.ndarray:
    """Apply degradation to a (N, H, W, 3) array. Returns same shape."""
    fn = _DISPATCH[degradation]
    return np.stack([fn(img, severity) for img in images])
