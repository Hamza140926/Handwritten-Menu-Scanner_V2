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

Pairing strategy (deliberately position-based, not text-based):
    - Split regions into two columns by x-coordinate gap (largest gap in
      sorted x-positions - works for a free-form 2-column layout without
      assuming fixed pixel thresholds, since photos vary in resolution).
    - Whichever column has more regions with a successfully extracted
      numeric price (postprocess.py's price_value) is treated as the
      price column; the other is the name column. This is data-driven
      rather than hardcoded "left=name", though in practice it will
      usually land that way for LTR-written menus.
    - Pair each name with its nearest price region by y-distance
      (nearest-neighbor, one-to-one). This is deliberately POSITION-based
      rather than TEXT-based: a price region whose text OCR'd to garbage
      (e.g. no digits recognized at all) still gets correctly linked to
      its item by position, rather than being mistaken for a second
      name/header just because its text didn't look like a price. This
      was confirmed necessary against a real scan where one price region
      recognized as unreadable garbage ('gdt .') still needed to end up
      attached to its item ('mocha'), not dropped or misclassified.
    - A name with no nearby price within a reasonable y-distance is
      treated as a category header (e.g. "Coffee", "Non Coffee") - this
      matches how real menus are actually laid out (confirmed against
      the same real scan).
    - A price with no nearby name is kept as an "orphan" and flagged for
      manual review rather than silently dropped, per the spec's
      correction-first philosophy (§2: build for correction, not
      perfection - never silently discard a signal the owner could fix).

This is a best-effort layout heuristic, not a guarantee - see spec §10
(open questions) re: whether a more robust layout model is ever needed.
Downstream (the review UI) should treat "is_category_header" and
"orphan" as hints, not ground truth.
"""

from preprocessing import preprocess_image
from detection import detect_text_regions, draw_regions_debug
from recognition import recognize_regions
from postprocess import process_recognition_results
from pairing import pair_regions_advanced, build_skeleton
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


def assemble_menu(processed_regions: list, max_y_distance: float = None) -> dict:
    """Turn postprocess.py's flat list of regions into a structured menu:
    category headers, named items with prices, and any orphaned prices
    that need manual review.
    
    Uses advanced tilted-line pairing algorithm from pairing.py for better
    accuracy than simple y-distance nearest-neighbor.
    
    Also calculates quality metrics to help identify low-quality scans.

    Returns a dict:
        "items":          list from pair_regions_advanced()
        "orphan_prices":  list from pair_regions_advanced()
        "quality_metrics": dict with:
            "total_regions": total text regions detected
            "items_with_prices": count of items with prices
            "category_headers": count of category headers (no price)
            "orphan_prices": count of unmatched prices
            "pairing_success_rate": % of items successfully paired with prices
            "avg_confidence": average recognition confidence (0-1)
            "low_confidence_items": count of items with confidence < threshold
            "ambiguous_prices": count of prices flagged as ambiguous
            "warnings": list of quality warnings
    """
    cfg = get_config().pipeline
    
    # Use advanced pairing algorithm (replaces simple y-distance pairing)
    items, orphan_prices = pair_regions_advanced(processed_regions)
    
    # Calculate quality metrics
    total_regions = len(processed_regions)
    items_with_prices = sum(1 for item in items if item["price_value"] is not None)
    category_headers = sum(1 for item in items if item["is_category_header"])
    orphan_count = len(orphan_prices)
    
    # Pairing success rate (excluding category headers)
    non_header_items = len(items) - category_headers
    pairing_success_rate = (items_with_prices / non_header_items * 100) if non_header_items > 0 else 0.0
    
    # Average confidence across all recognized text
    all_confidences = [r.get("confidence", 0.0) for r in processed_regions]
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    
    # Low confidence items
    low_confidence_items = sum(1 for r in processed_regions if r.get("confidence", 0.0) < cfg.min_confidence_warning)
    
    # Ambiguous prices
    ambiguous_prices = sum(1 for r in processed_regions if r.get("price_ambiguous", False))
    
    # Generate warnings
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
        "pairing_success_rate": round(pairing_success_rate, 1),
        "avg_confidence": round(avg_confidence, 3),
        "low_confidence_items": low_confidence_items,
        "ambiguous_prices": ambiguous_prices,
        "warnings": warnings
    }
    
    return {
        "items": items,
        "orphan_prices": orphan_prices,
        "quality_metrics": quality_metrics
    }


def assemble_from_skeleton(
    skeleton: list, recognized_by_box_id: dict, default_currency: str = None
) -> dict:
    """Join the geometry-only skeleton (pairing.build_skeleton, run right
    after detection) with recognized text/price (keyed by box_id, only
    present for non-noise boxes) into the final menu structure.

    This is the skeleton-first counterpart to assemble_menu(): structure
    (category/item/pairing) was already decided from geometry before
    recognition ran, so this function's only job is to drop recognized
    text and parsed price into the slots geometry already found - plus
    resolve "unresolved" boxes (see build_skeleton's docstring), the one
    case geometry couldn't settle on its own.

    Returns the same shape as assemble_menu(): {"items", "orphan_prices",
    "quality_metrics"}.
    """
    if default_currency is None:
        default_currency = get_config().postprocessing.default_currency
    cfg = get_config().pipeline

    items = []
    orphan_prices = []

    # Category headers
    for entry in skeleton:
        if entry["role"] != "category":
            continue
        rec = recognized_by_box_id.get(entry["box_id"])
        items.append({
            "name": rec["text"] if rec else "",
            "name_confidence": rec["confidence"] if rec else 0.0,
            "price_value": None, "price_raw": None, "price_ambiguous": False,
            "price_confidence": None, "currency": None, "currency_source": None,
            "is_category_header": True,
        })

    # Paired name<->price items
    seen_pairs = set()
    for entry in skeleton:
        if entry["role"] not in ("item_name", "item_price"):
            continue
        pair_key = tuple(sorted((entry["box_id"], entry["pair_id"])))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        if entry["role"] == "item_name":
            name_id, price_id = entry["box_id"], entry["pair_id"]
        else:
            name_id, price_id = entry["pair_id"], entry["box_id"]
        name_rec = recognized_by_box_id.get(name_id)
        price_rec = recognized_by_box_id.get(price_id)

        items.append({
            "name": name_rec["text"] if name_rec else "",
            "name_confidence": name_rec["confidence"] if name_rec else 0.0,
            "price_value": price_rec.get("price_value") if price_rec else None,
            "price_raw": price_rec.get("price_raw") if price_rec else None,
            "price_ambiguous": price_rec.get("price_ambiguous", False) if price_rec else False,
            "price_confidence": price_rec["confidence"] if price_rec else None,
            "currency": price_rec.get("currency") if price_rec else None,
            "currency_source": price_rec.get("currency_source") if price_rec else None,
            "is_category_header": False,
        })

    # Unresolved boxes: geometry found no pairing partner, so resolve using
    # recognized text now, same way the old pairing did for orphans.
    for entry in skeleton:
        if entry["role"] != "unresolved":
            continue
        rec = recognized_by_box_id.get(entry["box_id"])
        if rec is None:
            continue
        if rec.get("price_value") is not None:
            orphan_prices.append({
                "text": rec["text"], "confidence": rec["confidence"],
                "price_value": rec.get("price_value"), "price_raw": rec.get("price_raw"),
            })
        else:
            items.append({
                "name": rec["text"], "name_confidence": rec["confidence"],
                "price_value": None, "price_raw": None, "price_ambiguous": False,
                "price_confidence": None, "currency": None, "currency_source": None,
                "is_category_header": False,
                "needs_review": True,  # geometry never found a price pair for this item
            })

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


def interactive_detection(preprocessed_image: "np.ndarray", output_dir: str = ".") -> tuple:
    """
    Run detection with interactive review and retry capability.
    
    User sees visualization of detected boxes + skeleton classification,
    then can choose to continue or retry with different detection params.
    
    Args:
        preprocessed_image: BGR image from preprocessing
        output_dir: where to save visualization images
    
    Returns:
        (regions, skeleton) tuple for the accepted detection result
    """
    cfg = get_config().detection
    strategies = cfg.strategies
    current_strategy = "default"
    
    while True:
        # Apply current strategy params
        strategy_params = strategies[current_strategy]
        logger.info(f"Running detection with strategy: {current_strategy}", extra=strategy_params)
        
        # Run detection
        regions = detect_text_regions(
            preprocessed_image,
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
        
        # Build skeleton for classification
        skeleton = build_skeleton(regions)
        
        # Generate visualization
        vis_image = draw_regions_debug(preprocessed_image, regions, skeleton)
        vis_path = os.path.join(output_dir, "detection_preview.png")
        cv2.imwrite(vis_path, vis_image)
        
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
            return regions, skeleton
        elif choice == 'q':
            raise KeyboardInterrupt("User quit during detection review")
        elif choice == 'r':
            current_strategy = _prompt_strategy_choice(strategies, current_strategy)
        else:
            print("Invalid choice, assuming 'continue'")
            return regions, skeleton


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


def run_pipeline(image_path: str, default_currency: str = None) -> dict:
    """Run the full pipeline on a menu photo: preprocess -> detect ->
    recognize -> postprocess -> assemble into menu items.
    
    Validates input before processing and handles errors gracefully.
    Each stage can fail independently without crashing the entire pipeline.
    
    Memory optimized: clears intermediate results after each stage to
    reduce peak memory usage.

    Returns the dict from assemble_menu(), or an error dict with:
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
    
    # Stage 2: Detection
    try:
        regions = detect_text_regions(prep["image"])
        logger.debug("Detection complete", extra={"region_count": len(regions)})
        log_memory_usage("after_detection")
        
        # Memory optimization: clear preprocessing results, only keep regions
        del prep
        log_memory_usage("after_cleanup_prep")
        
        if not regions:
            logger.warning("No text regions detected")
            return {
                "items": [],
                "orphan_prices": [],
                "warning": "No text detected in image"
            }
    except DetectionError as e:
        logger.error("Detection stage failed", exc_info=True)
        raise

    # Stage 3: Skeleton (geometry-only classify + pair, BEFORE recognition)
    # regions[i] corresponds to skeleton[i] - "box_id" == index into regions.
    # Structure (category/item/pairing) is decided here from box geometry
    # alone, so it can't be corrupted by a bad OCR read downstream, and
    # noise boxes (illustrations, decorative headers) are identified now,
    # before the expensive recognition step ever has to look at them.
    try:
        skeleton = build_skeleton(regions)
        logger.debug("Skeleton built", extra={"box_count": len(skeleton)})
    except Exception as e:
        logger.error("Skeleton stage failed", exc_info=True)
        raise PipelineError(f"Skeleton building failed: {e}") from e

    # Stage 4: Recognition - only on boxes that survived noise filtering
    try:
        survivor_ids = [s["box_id"] for s in skeleton if s["role"] != "noise"]
        survivor_regions = [regions[i] for i in survivor_ids]

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
        del regions, survivor_regions
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
    args = parser.parse_args()

    image_path = args.image_path
    default_currency = args.currency or get_config().postprocessing.default_currency
    
    logger.info("Pipeline started", extra={"image_path": image_path, "currency": default_currency, "interactive": args.interactive})

    try:
        # For interactive mode, use special detection flow
        if args.interactive:
            # Preprocess + validate
            validated_path = validate_image_input(image_path)
            validated_currency = validate_currency(default_currency)
            prep = preprocess_image(str(validated_path))
            
            # Interactive detection with retry
            regions, skeleton = interactive_detection(prep["image"])
            
            if not regions:
                print("\nNo text detected. Exiting.")
                sys.exit(0)
            
            # Continue with rest of pipeline (recognition onwards)
            survivor_ids = [s["box_id"] for s in skeleton if s["role"] != "noise"]
            survivor_regions = [regions[i] for i in survivor_ids]
            recognized = recognize_regions(survivor_regions)
            processed = process_recognition_results(recognized, default_currency=validated_currency)
            recognized_by_box_id = dict(zip(survivor_ids, processed))
            menu = assemble_from_skeleton(skeleton, recognized_by_box_id, default_currency=validated_currency)
        else:
            # Non-interactive mode: use standard run_pipeline
            menu = run_pipeline(image_path, default_currency=default_currency)
        
        logger.info("Pipeline completed successfully", extra={
            "item_count": len(menu["items"]),
            "orphan_prices": len(menu["orphan_prices"])
        })

        # Display results
        print(f"\n{'='*50}\nMENU\n{'='*50}")
        for item in menu["items"]:
            if item["is_category_header"]:
                print(f"\n--- {item['name']} ---")
            else:
                price = item["price_value"] if item["price_value"] is not None else "?"
                flag = " (ambiguous)" if item["price_ambiguous"] else ""
                currency_str = item["currency"] or ""
                print(f"  {item['name']:25s} {price} {currency_str}{flag}")

        if menu["orphan_prices"]:
            print(f"\n{'='*50}\nUNMATCHED PRICES (need manual review)\n{'='*50}")
            for orphan in menu["orphan_prices"]:
                print(f"  text={orphan['text']!r}  price={orphan['price_value']}")
        
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