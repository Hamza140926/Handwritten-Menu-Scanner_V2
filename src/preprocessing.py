"""
Image preprocessing for handwritten menu scans.

Handles arbitrary uploaded photos/scans (unknown lighting, skew, resolution)
and prepares them for text detection AND recognition:
    - resize to a manageable working resolution
    - denoise
    - deskew (correct rotation)
    - normalize contrast / lighting (handles shadows, uneven lighting)

IMPORTANT (fix for geometry-vs-preprocessing mismatch):
    Detection/pairing (pairing.py) makes decisions from raw pixel geometry
    (column x-positions, row gaps, angles) using tolerances tuned in
    pixel units. Deskewing rotates every box's coordinates by a few
    degrees, which is enough to flip borderline classification/pairing
    votes on layouts that are already close to the vote threshold - this
    was confirmed empirically (same image, same algorithm, different
    results depending on whether detection ran before or after deskew).

    Denoise and CLAHE contrast normalization are pointwise operations -
    they don't move pixels around, so they don't affect box geometry.
    Deskew is the only step here that's geometrically unsafe for the
    detection/pairing stage.

    So: `preprocess_image()` now returns BOTH
        - "resized_raw": resized only (no denoise, no deskew, no CLAHE) -
          feed this into detection.detect_text_regions() for a stable,
          geometry-safe box set.
        - "image": fully preprocessed (denoised + deskewed) - use this
          for RECOGNITION crops only, via
          detection.recrop_for_recognition(), which re-projects each
          box's corners through "rotation_matrix" before cropping, so
          the crop still lines up correctly on the rotated image.
        - "rotation_matrix": the matrix used for deskew, or None if no
          rotation was applied (angle below the negligible threshold).
          Needed by recrop_for_recognition() to map box coordinates
          from "resized_raw" space into "image" (deskewed) space.

Usage:
    from preprocessing import preprocess_image

    result = preprocess_image("path/to/menu.jpg")
    # result["image"]         -> fully preprocessed BGR image (denoised, deskewed)
    # result["resized_raw"]   -> resized-only BGR image (use for detection)
    # result["gray"]          -> preprocessed grayscale image (deskewed, contrast-normalized)
    # result["angle"]         -> the deskew angle applied (degrees)
    # result["rotation_matrix"] -> 2x3 rotation matrix used for deskew, or None
"""

import cv2
import numpy as np
from logging_config import get_logger
from exceptions import PreprocessingError
from config import get_config

logger = get_logger(__name__)


def load_image(path: str) -> np.ndarray:
    """Load an image from disk. Raises a clear error if it fails."""
    logger.debug("Loading image", extra={"path": path})
    try:
        image = cv2.imread(path)
        if image is None:
            logger.error("Failed to load image", extra={"path": path})
            raise PreprocessingError(
                f"Could not load image at '{path}'. "
                "Check the file exists and is a valid image format."
            )
        h, w = image.shape[:2]
        logger.info("Image loaded successfully", extra={"path": path, "width": w, "height": h})
        return image
    except PreprocessingError:
        raise
    except Exception as e:
        logger.error("Unexpected error loading image", extra={"path": path, "error": str(e)})
        raise PreprocessingError(f"Failed to load image: {e}") from e


def resize_max_dimension(image: np.ndarray, max_dim: int = None) -> np.ndarray:
    """Resize image so its longer side is at most max_dim, preserving aspect ratio."""
    if max_dim is None:
        max_dim = get_config().preprocessing.max_dimension
    h, w = image.shape[:2]
    longer_side = max(h, w)
    if longer_side <= max_dim:
        return image
    scale = max_dim / longer_side
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def denoise(image: np.ndarray) -> np.ndarray:
    """Light denoising — removes speckle/noise from photographed paper without
    blurring handwriting strokes too much. Pointwise: does not move pixels,
    safe for anything that depends on box geometry."""
    cfg = get_config().preprocessing
    return cv2.fastNlMeansDenoisingColored(
        image, None,
        h=cfg.denoise_strength,
        hColor=cfg.denoise_strength,
        templateWindowSize=cfg.denoise_template_window,
        searchWindowSize=cfg.denoise_search_window
    )


def compute_skew_angle(gray: np.ndarray) -> float:
    """Estimate the rotation angle needed to deskew the image.

    Uses Canny edges + Hough line detection to find the dominant near-
    horizontal line angle (text baselines, ruled lines, note edges), which
    is far more robust on real photos than a whole-image minAreaRect —
    that approach can latch onto background/paper-edge shape and produce
    wildly wrong angles (e.g. ~90 degrees) on photos with visible
    background around the paper.

    A safety clamp ensures we never apply a large "correction" that isn't
    really there: real-world uploaded photos are rarely rotated more than
    a few degrees, so any estimate beyond max_skew_degrees is
    treated as noise and ignored (returns 0.0) rather than applied.
    """
    max_skew_degrees = get_config().preprocessing.max_skew_degrees
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80,
        minLineLength=gray.shape[1] // 8, maxLineGap=10
    )

    # Fallback with looser parameters if the strict pass finds nothing —
    # thin handwriting strokes / lower-contrast photos can produce fewer
    # edges than the strict pass expects.
    if lines is None or len(lines) == 0:
        logger.debug("No lines found with strict parameters, trying looser detection")
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=40,
            minLineLength=gray.shape[1] // 12, maxLineGap=15
        )

    if lines is None or len(lines) == 0:
        logger.warning("Skew detection failed: no lines found")
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue  # vertical line, not useful for horizontal-text skew
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only keep near-horizontal lines (text baselines), reject anything
        # closer to vertical (note edges, unrelated background lines)
        if abs(angle) <= max_skew_degrees:
            angles.append(angle)

    if not angles:
        logger.warning("Skew detection: no near-horizontal lines found")
        return 0.0

    estimated_angle = float(np.median(angles))

    # Final safety clamp — never trust an estimate outside a plausible range
    if abs(estimated_angle) > max_skew_degrees:
        logger.warning(
            "Skew angle outside plausible range, ignoring",
            extra={"estimated_angle": estimated_angle, "max_allowed": max_skew_degrees}
        )
        return 0.0

    logger.debug("Skew angle computed", extra={"angle": estimated_angle})
    return estimated_angle


def get_rotation_matrix(image_shape: tuple, angle: float) -> np.ndarray:
    """Build the 2x3 affine rotation matrix used for deskewing an image
    of the given shape by `angle` degrees around its center.

    Exposed separately (not just buried inside deskew()) so callers that
    need to re-project box coordinates - not full images - into the
    deskewed frame can reuse the exact same transform. See
    detection.recrop_for_recognition().
    """
    h, w = image_shape[:2]
    center = (w // 2, h // 2)
    return cv2.getRotationMatrix2D(center, angle, 1.0)


def deskew(image: np.ndarray, angle: float, rotation_matrix: np.ndarray = None) -> np.ndarray:
    """Rotate the image to correct the given skew angle.

    Accepts a precomputed rotation_matrix so callers building both the
    image and coordinate transforms (preprocess_image) only compute the
    matrix once.
    """
    if abs(angle) < 0.1:
        return image  # not worth rotating for negligible skew

    h, w = image.shape[:2]
    if rotation_matrix is None:
        rotation_matrix = get_rotation_matrix(image.shape, angle)
    rotated = cv2.warpAffine(
        image, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def normalize_contrast(gray: np.ndarray) -> np.ndarray:
    """Even out lighting/shadows using CLAHE (adaptive contrast).
    Pointwise: does not move pixels, safe for box geometry.

    This matters a lot for phone/scanner photos of paper menus, which
    often have uneven lighting or shadows across the page.
    """
    cfg = get_config().preprocessing
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=cfg.clahe_tile_size)
    return clahe.apply(gray)


def preprocess_image(path: str) -> dict:
    """Run the full preprocessing pipeline on an uploaded menu image.

    Returns a dict with:
        "image":           fully preprocessed BGR image (denoised, deskewed).
                            Use for RECOGNITION crops only (via
                            detection.recrop_for_recognition), not for
                            initial detection - see module docstring.
        "resized_raw":      resized-only BGR image (no denoise/deskew/CLAHE).
                            Feed this into detection.detect_text_regions()
                            for geometry-stable box detection.
        "gray":             preprocessed grayscale image (deskewed,
                            contrast-normalized).
        "angle":            the skew angle that was corrected, in degrees.
        "rotation_matrix":  2x3 matrix used to deskew, or None if no
                            rotation was applied (angle < 0.1 degrees).
                            Needed to re-project box coordinates from
                            "resized_raw" space into "image" space.

    Raises:
        PreprocessingError: If any preprocessing step fails
    """
    try:
        image = load_image(path)
        resized_raw = resize_max_dimension(image)

        # Detect skew BEFORE denoising — denoising blurs away the fine edges
        # (especially thin handwriting strokes) that skew detection relies on.
        # Running detection after denoise was causing silent failures (angle
        # always 0.0) on real photos.
        gray_for_skew = cv2.cvtColor(resized_raw, cv2.COLOR_BGR2GRAY)
        angle = compute_skew_angle(gray_for_skew)

        denoised = denoise(resized_raw)

        if abs(angle) >= 0.1:
            rotation_matrix = get_rotation_matrix(denoised.shape, angle)
            deskewed = deskew(denoised, angle, rotation_matrix)
        else:
            rotation_matrix = None
            deskewed = denoised

        gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
        gray = normalize_contrast(gray)

        return {
            "image": deskewed,
            "resized_raw": resized_raw,
            "gray": gray,
            "angle": angle,
            "rotation_matrix": rotation_matrix,
        }

    except PreprocessingError:
        raise
    except Exception as e:
        logger.exception("Preprocessing failed", extra={"path": path})
        raise PreprocessingError(f"Preprocessing pipeline failed: {e}") from e


if __name__ == "__main__":
    import sys

    from logging_config import setup_logging

    setup_logging(level="INFO")

    if len(sys.argv) != 2:
        logger.error("Missing image path argument")
        print("Usage: python preprocessing.py <path_to_image>")
        sys.exit(1)

    result = preprocess_image(sys.argv[1])
    logger.info("Preprocessing complete", extra={"angle": result['angle']})

    cv2.imwrite("preprocessed_output.png", result["image"])
    cv2.imwrite("preprocessed_gray.png", result["gray"])
    cv2.imwrite("resized_raw_output.png", result["resized_raw"])
    logger.info("Output saved", extra={
        "files": ["preprocessed_output.png", "preprocessed_gray.png", "resized_raw_output.png"]
    })