"""
Image preprocessing for handwritten menu scans.

Handles arbitrary uploaded photos/scans (unknown lighting, skew, resolution)
and prepares them for text detection:
    - resize to a manageable working resolution
    - denoise
    - deskew (correct rotation)
    - normalize contrast / lighting (handles shadows, uneven lighting)

Usage:
    from preprocessing import preprocess_image

    result = preprocess_image("path/to/menu.jpg")
    # result["image"] -> preprocessed OpenCV image (numpy array, BGR)
    # result["gray"]  -> grayscale version, useful for detection models
    # result["angle"] -> the deskew angle applied (degrees)
"""

import cv2
import numpy as np
from logging_config import get_logger
from exceptions import PreprocessingError

logger = get_logger(__name__)


MAX_DIMENSION = 2000  # cap the longer side so processing stays fast/consistent


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


def resize_max_dimension(image: np.ndarray, max_dim: int = MAX_DIMENSION) -> np.ndarray:
    """Resize image so its longer side is at most max_dim, preserving aspect ratio."""
    h, w = image.shape[:2]
    longer_side = max(h, w)
    if longer_side <= max_dim:
        return image
    scale = max_dim / longer_side
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def denoise(image: np.ndarray) -> np.ndarray:
    """Light denoising — removes speckle/noise from photographed paper without
    blurring handwriting strokes too much."""
    return cv2.fastNlMeansDenoisingColored(image, None, h=7, hColor=7,
                                            templateWindowSize=7, searchWindowSize=21)


MAX_SKEW_CORRECTION_DEGREES = 15.0  # real-world photo skew is rarely more than this;
                                     # anything beyond it is almost certainly a bad
                                     # estimate, not real rotation — clamp/skip instead


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
    a few degrees, so any estimate beyond MAX_SKEW_CORRECTION_DEGREES is
    treated as noise and ignored (returns 0.0) rather than applied.
    """
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
        if abs(angle) <= MAX_SKEW_CORRECTION_DEGREES:
            angles.append(angle)

    if not angles:
        logger.warning("Skew detection: no near-horizontal lines found")
        return 0.0

    estimated_angle = float(np.median(angles))

    # Final safety clamp — never trust an estimate outside a plausible range
    if abs(estimated_angle) > MAX_SKEW_CORRECTION_DEGREES:
        logger.warning(
            "Skew angle outside plausible range, ignoring",
            extra={"estimated_angle": estimated_angle, "max_allowed": MAX_SKEW_CORRECTION_DEGREES}
        )
        return 0.0

    logger.debug("Skew angle computed", extra={"angle": estimated_angle})
    return estimated_angle


def deskew(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate the image to correct the given skew angle."""
    if abs(angle) < 0.1:
        return image  # not worth rotating for negligible skew

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def normalize_contrast(gray: np.ndarray) -> np.ndarray:
    """Even out lighting/shadows using CLAHE (adaptive contrast).

    This matters a lot for phone/scanner photos of paper menus, which
    often have uneven lighting or shadows across the page.
    """
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_image(path: str) -> dict:
    """Run the full preprocessing pipeline on an uploaded menu image.

    Returns a dict with:
        "image": preprocessed BGR image (deskewed, denoised)
        "gray":  preprocessed grayscale image (deskewed, contrast-normalized)
        "angle": the skew angle that was corrected, in degrees
        
    Raises:
        PreprocessingError: If any preprocessing step fails
    """
    try:
        image = load_image(path)
        image = resize_max_dimension(image)

        # Detect skew BEFORE denoising — denoising blurs away the fine edges
        # (especially thin handwriting strokes) that skew detection relies on.
        # Running detection after denoise was causing silent failures (angle
        # always 0.0) on real photos.
        gray_for_skew = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        angle = compute_skew_angle(gray_for_skew)

        image = denoise(image)
        image = deskew(image, angle)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = normalize_contrast(gray)

        return {"image": image, "gray": gray, "angle": angle}
    
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
    logger.info("Output saved", extra={"files": ["preprocessed_output.png", "preprocessed_gray.png"]})