"""
Augmentations applied to rasterized handwriting-synthesis crops to close
some of the gap between "clean vector handwriting on white" and "ink on
paper, photographed with a phone camera" - which is what detection.py's
crops actually look like at inference time (see preprocessing.py: real
photos need denoising, deskewing, and contrast normalization in the first
place, which implies they start out noisy/uneven/skewed).

This does NOT close the full synthetic-to-real gap (no augmentation
pipeline reliably does) - it just means the model isn't training
exclusively on perfectly clean vector strokes, which would make it even
more brittle on real input than it already will be.
"""
import random
import numpy as np
import cv2


def tight_crop_to_ink(image: np.ndarray, padding: int = 8) -> np.ndarray:
    """Crop a rasterized (white background, dark strokes) image down to
    the bounding box of the actual ink, with padding - rendered SVGs
    otherwise carry a lot of surrounding whitespace that real detection
    crops don't have."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return image  # blank render - caller should discard this sample

    x, y, w, h = cv2.boundingRect(coords)
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(image.shape[1], x + w + padding)
    y1 = min(image.shape[0], y + h + padding)
    return image[y0:y1, x0:x1]


def add_paper_texture(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Overlay a mild uneven-lighting gradient, approximating the shadows
    and lighting variation preprocessing.py's CLAHE step is designed to
    correct for on real photos."""
    h, w = image.shape[:2]
    gradient_strength = rng.uniform(0.05, 0.2)
    angle = rng.uniform(0, 2 * np.pi)
    xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    gradient = np.cos(angle) * xx + np.sin(angle) * yy
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-6)
    shade = 1.0 - gradient_strength * gradient
    shade = shade[..., None] if image.ndim == 3 else shade
    out = np.clip(image.astype(np.float32) * shade, 0, 255).astype(np.uint8)
    return out


def add_noise(image: np.ndarray, rng: random.Random) -> np.ndarray:
    sigma = rng.uniform(2, 10)
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    out = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def random_blur(image: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        k = rng.choice([3, 3, 5])
        return cv2.GaussianBlur(image, (k, k), 0)
    return image


def random_rotation(image: np.ndarray, rng: random.Random, max_degrees=3.0) -> np.ndarray:
    angle = rng.uniform(-max_degrees, max_degrees)
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )


def jpeg_compress(image: np.ndarray, rng: random.Random) -> np.ndarray:
    quality = rng.randint(55, 90)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return image
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def brightness_contrast(image: np.ndarray, rng: random.Random) -> np.ndarray:
    alpha = rng.uniform(0.85, 1.15)  # contrast
    beta = rng.uniform(-15, 15)      # brightness
    return np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def photo_realism_pipeline(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Full augmentation chain applied to one crop. Order matters:
    geometric first, then photometric, then compression last (compression
    artifacts should sit on top of everything else, like a real photo)."""
    image = tight_crop_to_ink(image)
    image = random_rotation(image, rng)
    image = add_paper_texture(image, rng)
    image = brightness_contrast(image, rng)
    image = add_noise(image, rng)
    image = random_blur(image, rng)
    image = jpeg_compress(image, rng)
    return image
