"""
Text region detection for handwritten menu scans.

Finds each line/item of text on a (preprocessed) menu image and returns
cropped regions ready to feed into the recognition model.

Uses PaddleOCR's pretrained detector (DBNet-based) — detection only, no
recognition here. This handles free-form/messy layouts well since it
doesn't assume a clean row/column structure; it finds arbitrary text
regions wherever they are on the page.

Usage:
    from detection import detect_text_regions

    regions = detect_text_regions(image)
    # regions -> list of dicts, each with:
    #   "crop":  cropped BGR image of just that text region
    #   "box":   the 4 corner points of the region in the original image
    #   "y":     approximate vertical position (for reading-order sorting)

Note: PaddleOCR downloads its pretrained detection model weights the
first time it runs (small download, one-time, free — no account/API key
needed).
"""

import cv2
import numpy as np
import threading
from logging_config import get_logger

logger = get_logger(__name__)

_detector = None  # lazy-loaded singleton so the model loads once, not per call
_detector_lock = threading.Lock()  # thread-safe initialization


def _get_detector():
    """Load the PaddleOCR text detector once and reuse it across calls.

    Uses PaddleOCR 3.x's dedicated TextDetection module (detection-only —
    no recognition/orientation-classification overhead, which we don't
    need at this stage). This replaced the older PaddleOCR(det=, rec=)
    pipeline-style API from 2.x.

    Thread-safe: Multiple simultaneous calls will wait for initialization
    to complete rather than creating duplicate detector instances.

    Defaults to CPU. If you've installed paddlepaddle-gpu (matching your
    CUDA version) instead of the plain CPU build, pass device="gpu:0" to
    detect_text_regions() to use it.
    """
    global _detector
    if _detector is None:
        with _detector_lock:
            # Double-check locking pattern: another thread might have
            # initialized while we were waiting for the lock
            if _detector is None:
                from paddleocr import TextDetection
                # "mobile" model: smaller/faster, good fit for a 4GB GPU or CPU.
                # Swap to "PP-OCRv5_server_det" for higher accuracy if your
                # hardware handles it comfortably.
                #
                # enable_mkldnn=False works around a known bug in PaddlePaddle
                # 3.3.x's CPU inference backend (oneDNN/PIR executor) that throws
                # "NotImplementedError: ConvertPirAttribute2RuntimeAttribute not
                # support [...]" on CPU inference with MKL-DNN enabled (the
                # default). See: github.com/PaddlePaddle/Paddle/issues/77340
                logger.info("Initializing PaddleOCR text detector")
                _detector = TextDetection(model_name="PP-OCRv5_mobile_det", enable_mkldnn=False)
                logger.info("PaddleOCR text detector loaded successfully")
    return _detector


def _extract_polygons(result_item) -> list:
    """Pull the detected polygon list out of a PaddleOCR 3.x result object.

    PaddleOCR 3.x result objects support dict-style access to fields like
    'dt_polys', but the exact structure has shifted across point releases,
    so this tries a couple of known shapes and fails loudly with the raw
    object printed if none match — that's more useful for debugging than
    a silent empty result.
    """
    try:
        return list(result_item["dt_polys"])
    except (KeyError, TypeError):
        pass
    try:
        return list(result_item["res"]["dt_polys"])
    except (KeyError, TypeError):
        pass
    try:
        return list(result_item.json["res"]["dt_polys"])
    except (AttributeError, KeyError, TypeError):
        pass

    raise RuntimeError(
        "Could not find 'dt_polys' in the PaddleOCR result object. "
        f"Raw result for debugging: {result_item}"
    )


def _order_box_points(box: np.ndarray) -> np.ndarray:
    """Order a 4-point box as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = box.sum(axis=1)
    rect[0] = box[np.argmin(s)]
    rect[2] = box[np.argmax(s)]
    diff = np.diff(box, axis=1)
    rect[1] = box[np.argmin(diff)]
    rect[3] = box[np.argmax(diff)]
    return rect


def _crop_box(image: np.ndarray, box: np.ndarray, padding: int = 4) -> np.ndarray:
    """Crop a (possibly slightly rotated) box region out of the image using
    a perspective warp, so angled text lines are still extracted cleanly
    rather than just axis-aligned-bounding-box cropped (which would
    include extra neighboring content on tilted lines)."""
    rect = _order_box_points(box)
    (tl, tr, br, bl) = rect

    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))) + padding * 2
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))) + padding * 2
    width = max(width, 1)
    height = max(height, 1)

    destination = np.array([
        [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, destination)
    crop = cv2.warpPerspective(image, matrix, (width, height))
    return crop


def detect_text_regions(image: np.ndarray, min_box_area: int = 200) -> list:
    """Detect text regions in a preprocessed menu image.

    Args:
        image: preprocessed BGR image (e.g. from preprocessing.preprocess_image)
        min_box_area: discard detected boxes smaller than this (filters out
            noise/speckle false positives)

    Returns:
        List of dicts, sorted top-to-bottom then left-to-right
        (approximate natural reading order), each with:
            "crop": cropped BGR image of the text region
            "box":  the original 4 corner points in the source image
            "y":    approximate vertical center, used for sorting
    """
    logger.debug("Starting text detection", extra={"min_box_area": min_box_area})
    detector = _get_detector()
    output = detector.predict(input=image, batch_size=1)

    # predict() returns an iterable of one result per input image; we only
    # passed one image, so take the first (only) result.
    result_item = next(iter(output))
    raw_boxes = _extract_polygons(result_item)
    logger.debug("Raw detection complete", extra={"raw_boxes_count": len(raw_boxes)})

    regions = []
    for box in raw_boxes:
        box = np.array(box, dtype="float32")

        # Filter tiny/degenerate boxes
        area = cv2.contourArea(box)
        if area < min_box_area:
            continue

        crop = _crop_box(image, box)
        y_center = float(np.mean(box[:, 1]))
        x_center = float(np.mean(box[:, 0]))

        regions.append({
            "crop": crop,
            "box": box,
            "y": y_center,
            "x": x_center,
        })

    # Approximate natural reading order: sort primarily top-to-bottom,
    # then left-to-right within a similar vertical band. Free-form menus
    # won't always have perfectly aligned rows, so this is a best-effort
    # sort, not a guarantee — downstream (recognition + review UI) should
    # not hard-depend on perfect ordering.
    regions.sort(key=lambda r: (round(r["y"] / 20), r["x"]))
    
    logger.info("Text detection complete", extra={"regions_found": len(regions), "filtered_out": len(raw_boxes) - len(regions)})

    return regions


def draw_regions_debug(image: np.ndarray, regions: list) -> np.ndarray:
    """Draw detected region boxes on a copy of the image — useful for
    visually sanity-checking detection quality during development."""
    debug_image = image.copy()
    for i, region in enumerate(regions):
        box = region["box"].astype(int)
        cv2.polylines(debug_image, [box], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(debug_image, str(i), tuple(box[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return debug_image


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocessing import preprocess_image
    from logging_config import setup_logging
    
    setup_logging(level="INFO")

    if len(sys.argv) != 2:
        logger.error("Missing image path argument")
        print("Usage: python detection.py <path_to_image>")
        sys.exit(1)

    result = preprocess_image(sys.argv[1])
    image = result["image"]

    regions = detect_text_regions(image)

    debug_image = draw_regions_debug(image, regions)
    cv2.imwrite("detection_debug.png", debug_image)
    logger.info("Debug image saved", extra={"file": "detection_debug.png"})

    os.makedirs("detected_crops", exist_ok=True)
    for i, region in enumerate(regions):
        cv2.imwrite(f"detected_crops/region_{i:02d}.png", region["crop"])
    logger.info("Crops saved", extra={"count": len(regions), "directory": "detected_crops/"})