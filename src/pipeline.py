"""Ties preprocessing -> detection -> recognition -> postprocess together."""
"""
Ties preprocessing -> detection -> recognition -> postprocess together,
and assembles the result into actual menu items (name + price + currency)
ready for the owner review UI.

Why this file exists (and isn't just a thin glue script):
    detection.py finds text *regions* on the page - it has no concept of
    "menu item". In practice (confirmed against a real scanned menu, see
    docs/recognition-status-report.md), an item's name and its price
    come back as two separate regions in two separate columns, not one
    combined crop. Pairing those into actual items is a layout problem
    that belongs here, not in recognition.py or postprocess.py, which
    each only look at one region at a time.

Structure (category/item/pairing) is decided entirely from box geometry
by pairing.build_skeleton() right after detection, before recognition
ever runs - see pairing.py's module docstring for the classification/
pairing algorithm (shape-based noise filtering, 3-signal category vote,
Hungarian-algorithm name<->price pairing). This file's job is just to
wire that skeleton together with recognized text/price and produce the
final menu structure + quality metrics.

DETECTION-VS-PREPROCESSING SPLIT (important, read this before touching
Stage 2/3 below):
    Detection and skeleton-building run on preprocess_image()'s
    "resized_raw" - resized only, NOT denoised/deskewed. This was a
    deliberate fix: deskew rotation shifts every box's coordinates by
    enough to flip borderline classification/pairing votes in
    pairing.py, whose thresholds (COLUMN_X_TOLERANCE, angle checks,
    row-gap ratios) are pixel-based and were tuned against un-rotated
    geometry. Running detection on the deskewed image was silently
    corrupting skeleton results (categories misdetected, items dropping
    to "unresolved") even though detection itself is deterministic -
    confirmed by diffing skeleton roles box-by-box between the two
    inputs on a real menu photo.

    Recognition still benefits from the full preprocessing (denoise,
    deskew, CLAHE), since handwriting legibility is a pointwise/rotation
    concern, not a cross-box-geometry one. So after the skeleton is
    built from resized_raw boxes, detection.recrop_for_recognition() is
    used to re-crop the SAME boxes (by re-projecting their corners
    through the deskew rotation matrix) from the fully preprocessed
    image, before recognition runs on them. The skeleton's box_id
    indexing is untouched by this - only the "crop" pixels change.
"""

from preprocessing import preprocess_image
from detection import detect_text_regions, draw_regions_debug, recrop_for_recognition
from recognition import recognize_regions
from postprocess import process_recognition_results
from pairing import build_skeleton
from validation import validate_image_input, validate_currency, ValidationError
from exceptions import (
    PipelineError, PreprocessingError, DetectionError, 
    RecognitionError, PostprocessingError, AssemblyError
)
from optimization import warmup_gpu, optimize_for_throughput, log_memory_usage
from logging_config import get_logger
from config import get_config
import cv2
import os

logger = get_logger(__name__)

# Apply performance optimizations on module load
optimize_for_throughput()
warmup_gpu()


def split_columns(regions: list, gap_factor: float = None) -> tuple:
    """Split regions into two side-by-side columns by x-position.

    Finds the largest gap between consecutive x-positions (sorted) and
    splits there, but only if that gap is clearly bigger than the
    average spacing - otherwise this isn't really a two-column layout
    and everything is returned as a single column.

    Returns (left_column, right_column) - right_column is [] if no
    clear two-column split was found.
    """
    if gap_factor is None:
        gap_factor = get_config().pipeline.column_gap_factor
    if len(regions) < 2:
        return list(regions), []

    sorted_regions = sorted(regions, key=lambda r: r["x"])
    xs = [r["x"] for r in sorted_regions]
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]

    max_gap = max(gaps)
    if max_gap <= 0:
        return sorted_regions, []

    avg_gap = sum(gaps) / len(gaps)
    if max_gap <= avg_gap * gap_factor:
        return sorted_regions, []

    split_idx = gaps.index(max_gap)
    return sorted_regions[: split_idx + 1], sorted_regions[split_idx + 1 :]


def _price_hit_rate(column: list) -> float:
    """Fraction of regions in a column that have a successfully
    extracted numeric price. Used to decide which of the two columns
    from split_columns is the price column."""
    if not column:
        return 0.0
    hits = sum(1 for r in column if r.get("price_value") is not None)
    return hits / len(column)


def _write_pairing_debug_image(image, regions: list, skeleton: list, output_path: str) -> str:
    """Render the color-coded skeleton (categories/items/noise/unresolved
    + pairing lines) over the source image and save it. Thin wrapper
    around detection.draw_regions_debug so run_pipeline and
    interactive_detection share one implementation.

    `image` should be the SAME image `regions` was detected on
    (resized_raw), since "box" coordinates are drawn directly against it -
    passing the deskewed image here would misalign the overlay.

    Returns output_path for convenience (also logged).
    """
    debug_image = draw_regions_debug(image, regions, skeleton)
    cv2.imwrite(output_path, debug_image)
    logger.info("Pairing debug image written", extra={"path": output_path})
    return output_path


def identify_name_and_price_columns(left: list, right: list) -> tuple:

    """Decide which of the two columns from split_columns is the name
    column and which is the price column, based on which one actually
    contains more successfully-parsed prices - not by assuming a fixed
    left/right convention.

    Falls back to "right column is prices" (the common convention for
    LTR-written menus) if neither column has any parsed prices at all,
    since there's no signal to decide otherwise.
    """
    if not right:
        return left, []

    left_rate = _price_hit_rate(left)
    right_rate = _price_hit_rate(right)

    if right_rate >= left_rate:
        return left, right
    return right, left


def pair_items(
    name_column: list, price_column: list, max_y_distance: float = None
) -> tuple:
    """Pair each name region with its nearest (by y-distance) unclaimed
    price region.

    A name with no price within max_y_distance is treated as a category
    header. A price left unclaimed after all names are matched is
    returned separately as an "orphan" for manual review.

    Returns (items, orphan_prices):
        items: list of dicts, one per name region, each with:
            "name", "name_confidence"
            "price_value", "price_raw", "price_confidence", "price_ambiguous"
                (all None if no price was paired - category header)
            "currency", "currency_source" (from the paired price region,
                if any; None if no price was paired)
            "is_category_header": bool
        orphan_prices: list of dicts for price regions that never found
            a nearby name, each with "price_value", "price_raw", "text",
            "confidence" - surfaced so nothing is silently dropped.
    """
    if max_y_distance is None:
        max_y_distance = get_config().pipeline.max_y_distance
    sorted_names = sorted(name_column, key=lambda r: r["y"])
    unmatched_prices = sorted(price_column, key=lambda r: r["y"])

    items = []
    for name in sorted_names:
        best, best_dist = None, None
        for price in unmatched_prices:
            dist = abs(price["y"] - name["y"])
            if dist <= max_y_distance and (best is None or dist < best_dist):
                best, best_dist = price, dist

        if best is not None:
            unmatched_prices.remove(best)
            items.append({
                "name": name["text"],
                "name_confidence": name["confidence"],
                "price_value": best["price_value"],
                "price_raw": best["price_raw"],
                "price_ambiguous": best["price_ambiguous"],
                "price_confidence": best["confidence"],
                "currency": best["currency"],
                "currency_source": best["currency_source"],
                "is_category_header": False,
            })
        else:
            items.append({
                "name": name["text"],
                "name_confidence": name["confidence"],
                "price_value": None,
                "price_raw": None,
                "price_ambiguous": False,
                "price_confidence": None,
                "currency": None,
                "currency_source": None,
                "is_category_header": True,
            })

    orphan_prices = [
        {
            "text": p["text"],
            "confidence": p["confidence"],
            "price_value": p["price_value"],
            "price_raw": p["price_raw"],
        }
        for p in unmatched_prices
    ]

    return items, orphan_prices


def assemble_from_skeleton(
    skeleton: list, recognized_by_box_id: dict, default_currency: str = None
) -> dict:
    """Join the geometry-only skeleton (pairing.build_skeleton, run right
    after detection) with recognized text/price (keyed by box_id, only
    present for non-noise boxes) into the final menu structure.

    Structure (category/item/pairing) was already decided from geometry
    before recognition ran, so this function's only job is to drop
    recognized text and parsed price into the slots geometry already
    found - plus resolve "unresolved" boxes (see build_skeleton's
    docstring), the one case geometry couldn't settle on its own.

    Returns:
        {"items", "orphan_prices", "quality_metrics"}.
    """
    if default_currency is None:
        default_currency = get_config().postprocessing.default_currency
    cfg = get_config().pipeline

    # box_id is the index skeleton entries were built in (build_skeleton
    # iterates `for box in boxes` where boxes = [Box(i, r) for i, r in
    # enumerate(regions)]), and detect_text_regions already sorts regions
    # top-to-bottom / left-to-right before that - so box_id ascending is
    # already a decent proxy for reading order, same as sorting by
    # ycenter would give (which is what experiment2.py's print_report
    # does explicitly - box_id order gets us there without needing to
    # carry y-coordinates through the skeleton dicts).
    category_ids: list = []                  # box_id, in reading order
    category_payload: dict = {}              # box_id -> header item dict
    grouped_items: dict = {}                 # category box_id -> [(sort_key, item)]
    unassigned_items: list = []              # [(sort_key, item)] - no category above them
    orphan_prices = []
    seen_pairs = set()

    for entry in skeleton:
        role = entry["role"]

        if role == "category":
            box_id = entry["box_id"]
            rec = recognized_by_box_id.get(box_id)
            category_ids.append(box_id)
            category_payload[box_id] = {
                "name": rec["text"] if rec else "",
                "name_confidence": rec["confidence"] if rec else 0.0,
                "price_value": None, "price_raw": None, "price_ambiguous": False,
                "price_confidence": None, "currency": None, "currency_source": None,
                "is_category_header": True,
            }
            grouped_items.setdefault(box_id, [])

        elif role in ("item_name", "item_price"):
            pair_key = tuple(sorted((entry["box_id"], entry["pair_id"])))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            if role == "item_name":
                name_id, price_id = entry["box_id"], entry["pair_id"]
            else:
                name_id, price_id = entry["pair_id"], entry["box_id"]
            name_rec = recognized_by_box_id.get(name_id)
            price_rec = recognized_by_box_id.get(price_id)

            item = {
                "name": name_rec["text"] if name_rec else "",
                "name_confidence": name_rec["confidence"] if name_rec else 0.0,
                "price_value": price_rec.get("price_value") if price_rec else None,
                "price_raw": price_rec.get("price_raw") if price_rec else None,
                "price_ambiguous": price_rec.get("price_ambiguous", False) if price_rec else False,
                "price_confidence": price_rec["confidence"] if price_rec else None,
                "currency": price_rec.get("currency") if price_rec else None,
                "currency_source": price_rec.get("currency_source") if price_rec else None,
                "is_category_header": False,
            }
            sort_key = min(name_id, price_id)
            group_id = entry["group_id"]
            if group_id is not None:
                grouped_items.setdefault(group_id, []).append((sort_key, item))
            else:
                unassigned_items.append((sort_key, item))

        elif role == "unresolved":
            # geometry found no pairing partner - resolve using recognized
            # text now, same way the old pairing did for orphans.
            box_id = entry["box_id"]
            rec = recognized_by_box_id.get(box_id)
            if rec is None:
                continue
            if rec.get("price_value") is not None:
                orphan_prices.append({
                    "text": rec["text"], "confidence": rec["confidence"],
                    "price_value": rec.get("price_value"), "price_raw": rec.get("price_raw"),
                })
                continue
            item = {
                "name": rec["text"], "name_confidence": rec["confidence"],
                "price_value": None, "price_raw": None, "price_ambiguous": False,
                "price_confidence": None, "currency": None, "currency_source": None,
                "is_category_header": False,
                "needs_review": True,  # geometry never found a price pair for this item
            }
            group_id = entry["group_id"]
            if group_id is not None:
                grouped_items.setdefault(group_id, []).append((box_id, item))
            else:
                unassigned_items.append((box_id, item))

    # Assemble final ordered list: each category header immediately
    # followed by its own members (in reading order), mirroring
    # experiment2.py's terminal report. Items that never landed under any
    # category go last, as their own block - same convention
    # print_report used for "No category / unassigned".
    items = []
    for cat_id in category_ids:
        items.append(category_payload[cat_id])
        for _, item in sorted(grouped_items.get(cat_id, []), key=lambda t: t[0]):
            items.append(item)
    for _, item in sorted(unassigned_items, key=lambda t: t[0]):
        items.append(item)

    total_regions = len(skeleton)
    items_with_prices = sum(1 for i in items if i["price_value"] is not None)
    category_headers = sum(1 for i in items if i["is_category_header"])
    orphan_count = len(orphan_prices)
    non_header_items = len(items) - category_headers
    pairing_success_rate = (items_with_prices / non_header_items * 100) if non_header_items > 0 else 0.0

    all_confidences = [r["confidence"] for r in recognized_by_box_id.values()]
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    low_confidence_items = sum(
        1 for r in recognized_by_box_id.values() if r["confidence"] < cfg.min_confidence_warning
    )
    ambiguous_prices = sum(1 for r in recognized_by_box_id.values() if r.get("price_ambiguous"))
    noise_excluded = sum(1 for s in skeleton if s["role"] == "noise")

    warnings = []
    if avg_confidence < cfg.min_confidence_warning:
        warnings.append(f"Low average confidence ({avg_confidence:.1%}). Image quality may be poor.")
    if pairing_success_rate < 50:
        warnings.append(f"Low pairing success rate ({pairing_success_rate:.1f}%). Menu layout may be unusual.")
    orphan_ratio = orphan_count / total_regions if total_regions > 0 else 0
    if orphan_ratio > cfg.max_orphan_price_ratio:
        warnings.append(f"High orphan price ratio ({orphan_ratio:.1%}). Many prices couldn't be paired with items.")
    if low_confidence_items > total_regions * 0.3:
        warnings.append(f"{low_confidence_items} regions have low confidence. Manual review recommended.")
    if ambiguous_prices > 0:
        warnings.append(f"{ambiguous_prices} prices have ambiguous numbers. Review recommended.")

    quality_metrics = {
        "total_regions": total_regions,
        "items_with_prices": items_with_prices,
        "category_headers": category_headers,
        "orphan_prices": orphan_count,
        "noise_excluded": noise_excluded,
        "pairing_success_rate": round(pairing_success_rate, 1),
        "avg_confidence": round(avg_confidence, 3),
        "low_confidence_items": low_confidence_items,
        "ambiguous_prices": ambiguous_prices,
        "warnings": warnings,
    }

    return {"items": items, "orphan_prices": orphan_prices, "quality_metrics": quality_metrics}


def interactive_detection(prep: dict, output_dir: str = ".") -> tuple:
    """
    Run detection with interactive review and retry capability.
    
    User sees visualization of detected boxes + skeleton classification,
    then can choose to continue or retry with different detection params.
    
    Args:
        prep: the dict returned by preprocessing.preprocess_image() - needs
            "resized_raw" (for detection/skeleton) and "image" +
            "rotation_matrix" (for recognition re-cropping).
        output_dir: where to save visualization images
    
    Returns:
        (regions, skeleton) tuple for the accepted detection result.
        `regions`' "crop" fields point at the fully preprocessed image
        (deskewed/denoised), ready for recognition - "box"/"y"/"x" stay
        in resized_raw coordinate space, matching what the skeleton was
        built from.
    """
    cfg = get_config().detection
    strategies = cfg.strategies
    current_strategy = "default"

    detection_image = prep["resized_raw"]
    recognition_image = prep["image"]
    rotation_matrix = prep["rotation_matrix"]

    # build_skeleton's noise-shape check needs the source image's area -
    # computed once here since detection_image doesn't change across
    # strategy retries.
    image_area = float(detection_image.shape[0] * detection_image.shape[1])

    while True:
        # Apply current strategy params
        strategy_params = strategies[current_strategy]
        logger.info(f"Running detection with strategy: {current_strategy}", extra=strategy_params)
        
        # Run detection on the geometry-stable image
        regions = detect_text_regions(
            detection_image,
            min_box_area=strategy_params["min_box_area"]
        )
        
        if not regions:
            print("\n⚠ No text regions detected!")
            print("Try a different strategy or check image quality.\n")
            choice = input("Retry? (y/n): ").strip().lower()
            if choice != 'y':
                return regions, []
            current_strategy = _prompt_strategy_choice(strategies, current_strategy)
            continue
        
        # Build skeleton for classification (geometry only, on detection_image boxes)
        skeleton = build_skeleton(regions, image_area)
        
        # Generate visualization - drawn against detection_image since that's
        # what "box" coordinates are in.
        vis_path = os.path.join(output_dir, "detection_preview.png")
        _write_pairing_debug_image(detection_image, regions, skeleton, vis_path)
        
        # Show stats
        stats = {
            "total_boxes": len(regions),
            "categories": sum(1 for s in skeleton if s["role"] == "category"),
            "items": sum(1 for s in skeleton if s["role"] in ("item_name", "item_price")),
            "pairs": sum(1 for s in skeleton if s["role"] in ("item_name", "item_price") and s["pair_id"] is not None) // 2,
            "unresolved": sum(1 for s in skeleton if s["role"] == "unresolved"),
            "noise": sum(1 for s in skeleton if s["role"] == "noise"),
        }
        
        print(f"\n{'='*60}")
        print(f"DETECTION COMPLETE - Strategy: {current_strategy}")
        print(f"{'='*60}")
        print(f"Total boxes detected:    {stats['total_boxes']}")
        print(f"  Categories (headers):  {stats['categories']}")
        print(f"  Item boxes (paired):   {stats['items']} ({stats['pairs']} pairs)")
        print(f"  Unresolved (no pair):  {stats['unresolved']}")
        print(f"  Noise (excluded):      {stats['noise']}")
        print(f"\nVisualization saved: {vis_path}")
        print(f"\nColor coding:")
        print(f"  🟢 Green    = Item boxes (will be paired)")
        print(f"  🟠 Orange   = Category headers")
        print(f"  ⚪ Gray     = Noise (excluded from recognition)")
        print(f"  🔴 Red      = Unresolved (no pairing partner)")
        print(f"  🔵 Blue line = Pairing connections")
        print(f"{'='*60}\n")
        
        # Prompt for action
        print("Options:")
        print("  [c] Continue with recognition")
        print("  [r] Re-run detection with different strategy")
        print("  [q] Quit")
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == 'c':
            # Structure is locked in - now swap crops to the cleaner
            # preprocessed image for recognition, without touching the
            # geometry the skeleton was built from.
            recognition_regions = recrop_for_recognition(
                regions, recognition_image, rotation_matrix
            )
            return recognition_regions, skeleton
        elif choice == 'q':
            raise KeyboardInterrupt("User quit during detection review")
        elif choice == 'r':
            current_strategy = _prompt_strategy_choice(strategies, current_strategy)
        else:
            print("Invalid choice, assuming 'continue'")
            recognition_regions = recrop_for_recognition(
                regions, recognition_image, rotation_matrix
            )
            return recognition_regions, skeleton


def _prompt_strategy_choice(strategies: dict, current: str) -> str:
    """Prompt user to select a detection strategy."""
    print(f"\nAvailable detection strategies:")
    for i, (key, params) in enumerate(strategies.items(), 1):
        marker = " (current)" if key == current else ""
        print(f"  [{i}] {key:12s} - {params['description']}{marker}")
    print(f"  [Enter] Keep current ({current})")
    
    choice = input("\nSelect strategy number: ").strip()
    if not choice:
        return current
    
    try:
        idx = int(choice) - 1
        strategy_name = list(strategies.keys())[idx]
        return strategy_name
    except (ValueError, IndexError):
        print(f"Invalid choice, keeping {current}")
        return current


def run_pipeline(
    image_path: str,
    default_currency: str = None,
    debug_output_path: str = "pairing_debug.png",
) -> dict:
    """Run the full pipeline on a menu photo: preprocess -> detect ->
    recognize -> postprocess -> assemble into menu items.
    
    Validates input before processing and handles errors gracefully.
    Each stage can fail independently without crashing the entire pipeline.
    
    Memory optimized: clears intermediate results after each stage to
    reduce peak memory usage.

    Args:
        image_path: path to the menu photo.
        default_currency: currency code to assume when none is detected.
        debug_output_path: where to save the pairing debug image (boxes
            color-coded noise/category/item/unresolved, with lines drawn
            between paired name/price boxes). Written right after the
            skeleton is built, before the source image is freed. Pass
            None or "" to skip generating it.

    Returns the dict from assemble_from_skeleton(), plus
    "debug_image_path" (the path written to, or None if skipped), or an
    error dict with:
        "error": error type (validation, preprocessing, detection, etc.)
        "message": human-readable error message
        "items": empty list (for consistent response structure)
    
    Raises:
        ValidationError: If input validation fails (caller should handle)
        PipelineError: If any pipeline stage fails (caller should handle)
    """
    if default_currency is None:
        default_currency = get_config().postprocessing.default_currency
    
    logger.info("Starting pipeline", extra={"image_path": image_path, "currency": default_currency})
    
    # Validate inputs (raises ValidationError if invalid)
    validated_path = validate_image_input(image_path)
    validated_currency = validate_currency(default_currency)
    
    # Stage 1: Preprocessing
    try:
        log_memory_usage("before_preprocessing")
        prep = preprocess_image(str(validated_path))
        logger.debug("Preprocessing complete")
        log_memory_usage("after_preprocessing")
    except PreprocessingError as e:
        logger.error("Preprocessing stage failed", exc_info=True)
        raise
    
    # Stage 2: Detection - run on resized_raw (NOT deskewed/denoised), so
    # box geometry stays stable for the pixel-threshold-based skeleton
    # stage. See module docstring for why this split exists.
    try:
        detection_image = prep["resized_raw"]
        regions = detect_text_regions(detection_image)
        logger.debug("Detection complete", extra={"region_count": len(regions)})
        log_memory_usage("after_detection")

        # build_skeleton's noise-shape check needs the source image's area,
        # and (if requested) the debug image render needs the array itself
        # a little longer - hold both before prep gets freed below.
        image_area = float(detection_image.shape[0] * detection_image.shape[1])
        debug_image_source = detection_image if debug_output_path else None

        # Hang on to the recognition-quality image + rotation matrix for
        # Stage 4's recrop, then drop the rest of prep.
        recognition_image = prep["image"]
        rotation_matrix = prep["rotation_matrix"]
        del prep
        log_memory_usage("after_cleanup_prep")
        
        if not regions:
            logger.warning("No text regions detected")
            return {
                "items": [],
                "orphan_prices": [],
                "warning": "No text detected in image",
                "debug_image_path": None,
            }
    except DetectionError as e:
        logger.error("Detection stage failed", exc_info=True)
        raise

    # Stage 3: Skeleton (geometry-only classify + pair, BEFORE recognition)
    # regions[i] corresponds to skeleton[i] - "box_id" == index into regions.
    # Structure (category/item/pairing) is decided here from box geometry
    # alone (on the geometry-stable detection_image boxes), so it can't be
    # corrupted by a bad OCR read downstream, and noise boxes (illustrations,
    # decorative headers) are identified now, before the expensive
    # recognition step ever has to look at them.
    try:
        skeleton = build_skeleton(regions, image_area)
        logger.debug("Skeleton built", extra={"box_count": len(skeleton)})

        if debug_output_path:
            _write_pairing_debug_image(debug_image_source, regions, skeleton, debug_output_path)

        del debug_image_source
    except Exception as e:
        logger.error("Skeleton stage failed", exc_info=True)
        raise PipelineError(f"Skeleton building failed: {e}") from e

    # Stage 4: Recognition - only on boxes that survived noise filtering.
    # Re-crop survivors from the fully preprocessed (denoised/deskewed)
    # image now that structure is locked in - recognition benefits from
    # the cleaner pixels, and this can't disturb the skeleton since
    # box_id/role/pair_id/group_id were already decided in Stage 3.
    try:
        survivor_ids = [s["box_id"] for s in skeleton if s["role"] != "noise"]
        survivor_regions = [regions[i] for i in survivor_ids]
        survivor_regions = recrop_for_recognition(
            survivor_regions, recognition_image, rotation_matrix
        )

        recognized = recognize_regions(survivor_regions)
        logger.debug("Recognition complete", extra={
            "recognized_count": len(recognized),
            "noise_skipped": len(regions) - len(survivor_regions),
        })
        log_memory_usage("after_recognition")

        # Memory optimization: crops no longer needed after recognition,
        # for survivors and noise boxes alike
        for region in regions:
            if "crop" in region:
                del region["crop"]
        del regions, survivor_regions, recognition_image
        log_memory_usage("after_cleanup_crops")

    except RecognitionError as e:
        logger.error("Recognition stage failed", exc_info=True)
        raise
    
    # Stage 5: Postprocessing (price/currency parsing on recognized text)
    try:
        processed = process_recognition_results(recognized, default_currency=validated_currency)
        logger.debug("Postprocessing complete")

        # Re-attach box_id so assembly can join back against the skeleton -
        # recognize_regions/process_recognition_results don't carry custom
        # keys through, but survivor_ids is in the same order as the
        # regions that were fed into recognize_regions.
        recognized_by_box_id = dict(zip(survivor_ids, processed))

        del recognized, processed
        
    except Exception as e:
        logger.error("Postprocessing stage failed", exc_info=True)
        raise PostprocessingError(f"Postprocessing failed: {e}") from e
    
    # Stage 6: Assembly - join skeleton (structure) with recognized text/price
    try:
        menu = assemble_from_skeleton(skeleton, recognized_by_box_id, default_currency=validated_currency)
        menu["debug_image_path"] = debug_output_path if debug_output_path else None
        logger.info("Pipeline complete", extra={
            "items": len(menu["items"]),
            "orphans": len(menu["orphan_prices"])
        })
        log_memory_usage("after_assembly")

        del skeleton, recognized_by_box_id

        return menu
    except Exception as e:
        logger.error("Assembly stage failed", exc_info=True)
        raise AssemblyError(f"Menu assembly failed: {e}") from e


def _confidence_band(confidence: float) -> str:
    """Map a 0-1 confidence score to the same tiering the UI will use for
    color-coding (green/yellow/orange/red), so the CLI output and the
    eventual UI agree on what "needs review" means.

        >= 0.80        green
        0.50 - 0.80    yellow
        0.20 - 0.50    orange
        < 0.20         red
    """
    if confidence is None:
        return ""
    pct = confidence * 100
    if pct >= 80:
        band = "green"
    elif pct >= 50:
        band = "yellow"
    elif pct >= 20:
        band = "orange"
    else:
        band = "red"
    return f"({pct:.0f}% {band})"


if __name__ == "__main__":
    import sys
    import argparse
    from logging_config import setup_logging
    
    # Set up logging at application startup
    setup_logging(level="INFO")
    logger = get_logger(__name__)

    parser = argparse.ArgumentParser(description="Handwritten menu scanner pipeline")
    parser.add_argument("image_path", help="Path to menu image")
    parser.add_argument("currency", nargs='?', default=None, help="Currency code (TND, EUR)")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Interactive mode: review detection before continuing")
    parser.add_argument("--debug-out", default=None,
                       help="Path to write the pairing debug image (boxes color-coded "
                            "noise/category/item/unresolved, with name<->price pairing lines). "
                            "Defaults to '<image_stem>_pairing_debug.png'.")
    parser.add_argument("--no-debug-image", action="store_true",
                       help="Skip generating the pairing debug image entirely.")
    args = parser.parse_args()

    image_path = args.image_path
    default_currency = args.currency or get_config().postprocessing.default_currency

    if args.no_debug_image:
        debug_output_path = None
    elif args.debug_out:
        debug_output_path = args.debug_out
    else:
        debug_output_path = f"{os.path.splitext(os.path.basename(image_path))[0]}_pairing_debug.png"
    
    logger.info("Pipeline started", extra={"image_path": image_path, "currency": default_currency, "interactive": args.interactive})

    try:
        # For interactive mode, use special detection flow
        if args.interactive:
            # Preprocess + validate
            validated_path = validate_image_input(image_path)
            validated_currency = validate_currency(default_currency)
            prep = preprocess_image(str(validated_path))
            
            # Interactive detection with retry (handles image_area internally,
            # runs detection on resized_raw for stable geometry, and already
            # writes a pairing debug image - "detection_preview.png" - on
            # every strategy attempt, not just the accepted one)
            regions, skeleton = interactive_detection(prep)
            
            if not regions:
                print("\nNo text detected. Exiting.")
                sys.exit(0)
            
            # regions' crops already point at the preprocessed image
            # (interactive_detection re-crops on accept) - continue with
            # recognition onwards.
            survivor_ids = [s["box_id"] for s in skeleton if s["role"] != "noise"]
            survivor_regions = [regions[i] for i in survivor_ids]
            recognized = recognize_regions(survivor_regions)
            processed = process_recognition_results(recognized, default_currency=validated_currency)
            recognized_by_box_id = dict(zip(survivor_ids, processed))
            menu = assemble_from_skeleton(skeleton, recognized_by_box_id, default_currency=validated_currency)
            menu["debug_image_path"] = os.path.join(".", "detection_preview.png")
        else:
            # Non-interactive mode: use standard run_pipeline
            menu = run_pipeline(image_path, default_currency=default_currency, debug_output_path=debug_output_path)
        
        logger.info("Pipeline completed successfully", extra={
            "item_count": len(menu["items"]),
            "orphan_prices": len(menu["orphan_prices"])
        })

        # Display results
        print(f"\n{'='*50}\nMENU\n{'='*50}")
        for item in menu["items"]:
            name_conf = _confidence_band(item.get("name_confidence"))
            if item["is_category_header"]:
                print(f"\n--- {item['name']} {name_conf} ---")
            else:
                price = item["price_value"] if item["price_value"] is not None else "?"
                flag = " (ambiguous)" if item["price_ambiguous"] else ""
                currency_str = item["currency"] or ""
                price_conf = _confidence_band(item.get("price_confidence"))
                print(f"  {item['name']:25s} {name_conf:16s} {price} {currency_str} {price_conf}{flag}")

        if menu["orphan_prices"]:
            print(f"\n{'='*50}\nUNMATCHED PRICES (need manual review)\n{'='*50}")
            for orphan in menu["orphan_prices"]:
                print(f"  text={orphan['text']!r}  price={orphan['price_value']}")

        if menu.get("debug_image_path"):
            print(f"\nPairing debug image: {menu['debug_image_path']}")
        
        # Display quality metrics
        metrics = menu.get("quality_metrics", {})
        if metrics:
            print(f"\n{'='*50}\nQUALITY METRICS\n{'='*50}")
            print(f"Total Regions Detected:    {metrics['total_regions']}")
            print(f"Items with Prices:         {metrics['items_with_prices']}")
            print(f"Category Headers:          {metrics['category_headers']}")
            print(f"Orphan Prices:             {metrics['orphan_prices']}")
            print(f"Pairing Success Rate:      {metrics['pairing_success_rate']}%")
            print(f"Average Confidence:        {metrics['avg_confidence']:.1%}")
            print(f"Low Confidence Items:      {metrics['low_confidence_items']}")
            print(f"Ambiguous Prices:          {metrics['ambiguous_prices']}")
            
            if metrics["warnings"]:
                print(f"\n{'='*50}\nWARNINGS\n{'='*50}")
                for warning in metrics["warnings"]:
                    print(f"⚠  {warning}")
    
    except ValidationError as e:
        logger.error("Validation failed", extra={"error": str(e)})
        print(f"\nValidation Error: {e}")
        sys.exit(1)
    
    except PreprocessingError as e:
        logger.error("Preprocessing failed", extra={"error": str(e)})
        print(f"\nPreprocessing Error: {e}")
        print("The image could not be preprocessed. Check if the file is corrupt.")
        sys.exit(1)
    
    except DetectionError as e:
        logger.error("Detection failed", extra={"error": str(e)})
        print(f"\nDetection Error: {e}")
        print("Text detection failed. The image may not contain readable text.")
        sys.exit(1)
    
    except RecognitionError as e:
        logger.error("Recognition failed", extra={"error": str(e)})
        print(f"\nRecognition Error: {e}")
        print("Handwriting recognition failed. Try a clearer image.")
        sys.exit(1)
    
    except (PostprocessingError, AssemblyError) as e:
        logger.error("Pipeline stage failed", extra={"error": str(e)})
        print(f"\nPipeline Error: {e}")
        sys.exit(1)
    
    except Exception as e:
        logger.exception("Unexpected pipeline failure", extra={"image_path": image_path})
        print(f"\nUnexpected Error: {e}")
        print("An unexpected error occurred. Check the logs for details.")
        sys.exit(1)