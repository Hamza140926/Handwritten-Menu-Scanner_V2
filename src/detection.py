"""
Text region detection for handwritten menu scans.

Finds each line/item of text on a (preprocessed) menu image and returns
cropped regions ready to feed into the recognition model.

Uses PaddleOCR's pretrained detector (DBNet-based) — detection only, no
recognition here. This handles free-form/messy layouts well since it
doesn't assume a clean row/column structure; it finds arbitrary text
regions wherever they are on the page.

IMPORTANT: detect_text_regions() should be called on the resized-but-not
deskewed image (preprocessing.preprocess_image()'s "resized_raw"), not the
fully preprocessed one. Deskew rotation shifts box coordinates enough to
flip borderline classification/pairing decisions downstream in pairing.py
(confirmed empirically on real menu photos). If you also want the
denoised/deskewed image's cleaner pixels for recognition, use
recrop_for_recognition() below to re-crop the SAME boxes from that image
without re-running detection or changing the geometry the skeleton was
built from.

Usage:
    from detection import detect_text_regions, recrop_for_recognition

    regions = detect_text_regions(resized_raw_image)
    # ... build skeleton from `regions` here ...
    recognition_regions = recrop_for_recognition(
        regions, deskewed_image, rotation_matrix
    )
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
from exceptions import DetectionError
from config import get_config

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
    
    Raises:
        DetectionError: If model loading fails
    """
    global _detector
    if _detector is None:
        with _detector_lock:
            # Double-check locking pattern: another thread might have
            # initialized while we were waiting for the lock
            if _detector is None:
                try:
                    from paddleocr import TextDetection
                    cfg = get_config().detection
                    # "mobile" model: smaller/faster, good fit for a 4GB GPU or CPU.
                    # Swap to "PP-OCRv5_server_det" for higher accuracy if your
                    # hardware handles it comfortably.
                    #
                    # enable_mkldnn=False works around a known bug in PaddlePaddle
                    # 3.3.x's CPU inference backend (oneDNN/PIR executor) that throws
                    # "NotImplementedError: ConvertPirAttribute2RuntimeAttribute not
                    # support [...]" on CPU inference with MKL-DNN enabled (the
                    # default). See: github.com/PaddlePaddle/Paddle/issues/77340
                    logger.info("Initializing PaddleOCR text detector", extra={"model": cfg.model_name})
                    _detector = TextDetection(model_name=cfg.model_name, enable_mkldnn=cfg.enable_mkldnn)
                    logger.info("PaddleOCR text detector loaded successfully")
                except Exception as e:
                    logger.exception("Failed to load PaddleOCR detector")
                    raise DetectionError(f"Failed to initialize text detector: {e}") from e
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


def detect_text_regions(image: np.ndarray, min_box_area: int = None) -> list:
    """Detect text regions in an image.

    IMPORTANT: pass preprocessing.preprocess_image()'s "resized_raw" here,
    not "image" (the deskewed/denoised one) - see module docstring for why.

    Args:
        image: BGR image, resized but NOT deskewed/denoised (e.g.
            preprocess_image()["resized_raw"])
        min_box_area: discard detected boxes smaller than this (filters out
            noise/speckle false positives)

    Returns:
        List of dicts, sorted top-to-bottom then left-to-right
        (approximate natural reading order), each with:
            "crop": cropped BGR image of the text region
            "box":  the original 4 corner points in the source image
            "y":    approximate vertical center, used for sorting
            
    Raises:
        DetectionError: If detection fails
    """
    try:
        cfg = get_config().detection
        if min_box_area is None:
            min_box_area = cfg.min_box_area
        
        logger.debug("Starting text detection", extra={"min_box_area": min_box_area})
        detector = _get_detector()
        
        # Batch size from config
        output = detector.predict(input=image, batch_size=cfg.batch_size)

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
        row_tolerance = cfg.row_tolerance
        regions.sort(key=lambda r: (round(r["y"] / row_tolerance), r["x"]))
        
        logger.info("Text detection complete", extra={"regions_found": len(regions), "filtered_out": len(raw_boxes) - len(regions)})

        return regions
    
    except DetectionError:
        raise
    except Exception as e:
        logger.exception("Text detection failed")
        raise DetectionError(f"Detection pipeline failed: {e}") from e


def recrop_for_recognition(
    regions: list, recognition_image: np.ndarray, rotation_matrix: np.ndarray = None
) -> list:
    """Re-crop each detected region's "crop" from a differently-processed
    version of the image (e.g. denoised + deskewed), for use in
    recognition, WITHOUT touching the original "box" coordinates that the
    skeleton (pairing.build_skeleton) was built from.

    Why this exists: detection must run on geometry-stable pixels
    (preprocess_image()["resized_raw"]) so pairing.py's pixel-based
    thresholds behave consistently. But recognition benefits from the
    denoised/deskewed/contrast-normalized image. This function bridges
    the two: it re-projects each box's corner points through the same
    rotation used to produce `recognition_image` (if any), then re-crops
    from there - so the crop content matches the cleaner image while the
    "box" field used for geometry stays exactly as detected.

    Args:
        regions: output of detect_text_regions() (run on resized_raw)
        recognition_image: the fully preprocessed image, e.g.
            preprocess_image()["image"] (denoised + deskewed)
        rotation_matrix: preprocess_image()["rotation_matrix"] - the 2x3
            affine matrix used to deskew `recognition_image` relative to
            resized_raw. Pass None if no rotation was applied (angle was
            negligible), in which case box coordinates are assumed to
            already line up (only pointwise ops - denoise/CLAHE - were
            applied).

    Returns:
        A new list of region dicts (same length/order as `regions`),
        each with an updated "crop" pointing at the recognition image.
        "box", "y", "x" are left untouched so callers can still match
        these back to the original skeleton by list position / box_id.
    """
    updated = []
    for region in regions:
        box = region["box"]
        if rotation_matrix is not None:
            pts = np.array([box], dtype="float32")  # shape (1, 4, 2) for cv2.transform
            transformed_box = cv2.transform(pts, rotation_matrix)[0]
        else:
            transformed_box = box

        new_region = dict(region)
        new_region["crop"] = _crop_box(recognition_image, transformed_box)
        updated.append(new_region)

    return updated


def draw_regions_debug(image: np.ndarray, regions: list, skeleton: list = None) -> np.ndarray:
    """Draw detected region boxes on a copy of the image with optional
    skeleton classification overlay (categories, items, noise, pairings).
    
    Args:
        image: source image (BGR) - should be the SAME image regions'
            "box" coordinates were detected on (i.e. resized_raw), or the
            drawn boxes will be misaligned.
        regions: list of region dicts from detect_text_regions
        skeleton: optional skeleton from pairing.build_skeleton() - if
                  provided, boxes are color-coded by role and pairing
                  lines are drawn
    
    Color coding when skeleton provided:
        - Green: item boxes (will be paired)
        - Orange: category headers
        - Gray: noise (excluded from recognition)
        - Red: unresolved (no pair found)
        - Blue lines: pairing connections, name's right edge -> price's
          left edge
    """
    debug_image = image.copy()
    
    if skeleton is None:
        # Simple mode: just green boxes with numbers
        for i, region in enumerate(regions):
            box = region["box"].astype(int)
            cv2.polylines(debug_image, [box], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(debug_image, str(i), tuple(box[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        # Enhanced mode: color-code by role and draw pairing lines
        # First pass: draw pairing lines (background layer). Each pair
        # produces two skeleton entries pointing at each other (one
        # "item_name", one "item_price") - only draw from the
        # "item_name" side, or the pair gets drawn twice: once correctly
        # (name's right edge -> price's left edge) and once backwards
        # (price's right edge -> name's left edge), which is a visibly
        # different, wrong line, not just a redundant redraw.
        for entry in skeleton:
            if entry["role"] != "item_name" or entry["pair_id"] is None:
                continue
            name_box = regions[entry["box_id"]]["box"]
            price_box = regions[entry["pair_id"]]["box"]
            x1, y1 = int(name_box[:, 0].max()), int((name_box[:, 1].min() + name_box[:, 1].max()) / 2)
            x2, y2 = int(price_box[:, 0].min()), int((price_box[:, 1].min() + price_box[:, 1].max()) / 2)
            cv2.line(debug_image, (x1, y1), (x2, y2), (255, 100, 0), 2)
        
        # Second pass: draw boxes color-coded by role
        for i, entry in enumerate(skeleton):
            box = regions[entry["box_id"]]["box"].astype(int)
            role = entry["role"]
            
            if role == "noise":
                color, thickness = (150, 150, 150), 2  # gray
            elif role == "category":
                color, thickness = (0, 140, 255), 3    # orange
            elif role == "unresolved":
                color, thickness = (0, 0, 255), 3      # red
            else:  # item_name or item_price
                color, thickness = (0, 255, 0), 2       # green
            
            cv2.polylines(debug_image, [box], isClosed=True, color=color, thickness=thickness)
            cv2.putText(debug_image, str(entry["box_id"]), tuple(box[0]),
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
    image = result["resized_raw"]

    regions = detect_text_regions(image)

    debug_image = draw_regions_debug(image, regions)
    cv2.imwrite("detection_debug.png", debug_image)
    logger.info("Debug image saved", extra={"file": "detection_debug.png"})

    os.makedirs("detected_crops", exist_ok=True)
    for i, region in enumerate(regions):
        cv2.imwrite(f"detected_crops/region_{i:02d}.png", region["crop"])
    logger.info("Crops saved", extra={"count": len(regions), "directory": "detected_crops/"})