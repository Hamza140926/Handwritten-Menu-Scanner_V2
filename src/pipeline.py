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
from detection import detect_text_regions
from recognition import recognize_regions
from postprocess import process_recognition_results
from validation import validate_image_input, validate_currency, ValidationError
from exceptions import (
    PipelineError, PreprocessingError, DetectionError, 
    RecognitionError, PostprocessingError, AssemblyError
)
from optimization import warmup_gpu, optimize_for_throughput, log_memory_usage
from logging_config import get_logger
from config import get_config

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

    Returns a dict:
        "items":          list from pair_items()
        "orphan_prices":  list from pair_items()
    """
    if max_y_distance is None:
        max_y_distance = get_config().pipeline.max_y_distance
    left, right = split_columns(processed_regions)
    name_column, price_column = identify_name_and_price_columns(left, right)
    items, orphan_prices = pair_items(name_column, price_column, max_y_distance)
    return {"items": items, "orphan_prices": orphan_prices}


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
    
    # Stage 3: Recognition
    try:
        recognized = recognize_regions(regions)
        logger.debug("Recognition complete")
        log_memory_usage("after_recognition")
        
        # Memory optimization: clear crop images from regions (no longer needed)
        for region in regions:
            if "crop" in region:
                del region["crop"]
        del regions
        log_memory_usage("after_cleanup_crops")
        
    except RecognitionError as e:
        logger.error("Recognition stage failed", exc_info=True)
        raise
    
    # Stage 4: Postprocessing
    try:
        processed = process_recognition_results(recognized, default_currency=validated_currency)
        logger.debug("Postprocessing complete")
        
        # Memory optimization: clear intermediate recognition results
        del recognized
        
    except Exception as e:
        logger.error("Postprocessing stage failed", exc_info=True)
        raise PostprocessingError(f"Postprocessing failed: {e}") from e
    
    # Stage 5: Assembly
    try:
        menu = assemble_menu(processed)
        logger.info("Pipeline complete", extra={
            "items": len(menu["items"]),
            "orphans": len(menu["orphan_prices"])
        })
        log_memory_usage("after_assembly")
        
        # Memory optimization: clear processed regions (final result is in menu)
        del processed
        
        return menu
    except Exception as e:
        logger.error("Assembly stage failed", exc_info=True)
        raise AssemblyError(f"Menu assembly failed: {e}") from e


if __name__ == "__main__":
    import sys
    from logging_config import setup_logging
    
    # Set up logging at application startup
    setup_logging(level="INFO")
    logger = get_logger(__name__)

    if len(sys.argv) < 2:
        logger.error("Missing image path argument")
        print("Usage: python pipeline.py <path_to_image> [TND|EUR]")
        sys.exit(1)

    image_path = sys.argv[1]
    default_currency = sys.argv[2] if len(sys.argv) > 2 else get_config().postprocessing.default_currency
    
    logger.info("Pipeline started", extra={"image_path": image_path, "currency": default_currency})

    try:
        menu = run_pipeline(image_path, default_currency=default_currency)
        
        logger.info("Pipeline completed successfully", extra={
            "item_count": len(menu["items"]),
            "orphan_prices": len(menu["orphan_prices"])
        })

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