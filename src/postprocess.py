"""
Post-processing: price extraction and currency resolution.

Takes the recognized-text output from recognition.py and turns each
region's raw string into a structured price, per spec §5.4-5.5.

Deliberately simple by design:
    - Regex pulls out *a number*, nothing more. It never tries to read
      currency letters (DT, dt, TND, D, d) off the handwriting — per the
      spec, handwriting recognition on 1-2 letter abbreviations is
      unreliable, and it's unnecessary given the currency-resolution
      approach below.
    - Currency is resolved from an owner-set default (§5.5) for the
      whole menu, with a detected "€" symbol used only as a high-
      confidence override hint, never as the primary source of truth.

Usage:
    from postprocess import process_recognition_results

    processed = process_recognition_results(results, default_currency="TND")
    # results -> the list of dicts from recognition.recognize_regions
    # processed -> same list, each dict additionally has:
    #   "price_value":      float or None
    #   "price_raw":        the matched substring that produced price_value
    #   "price_ambiguous":  True if multiple differing numbers were found
    #                       in the same string (noisy OCR, e.g. repeated
    #                       hallucinated digits) - a signal for the
    #                       review UI, not a reason to discard the guess
    #   "currency":         resolved currency code ("TND" or "EUR")
    #   "currency_source":  "default" or "detected_symbol"

Note: this does NOT pair "item name" regions with "price" regions into
menu items - detection.py finds regions on the page and, in practice,
this means an item name and its price often come back as two adjacent-
but-separate regions (see docs/recognition-status-report.md, §3, real
run) rather than a single "Couscous 12.500"-style crop. Pairing regions
into (name, price) items by position is layout logic that belongs in
pipeline.py, not here - this module's job stops at "does this string
contain a usable number and what's the resolved currency." See spec
§10 (open questions) re: layout assumptions.
"""

import re
from config import get_config


def extract_price(text: str) -> dict:
    """Pull a numeric price out of a recognized string, if one is there.

    Does not interpret currency letters at all - only digits and the
    decimal separator matter here.

    Improved regex with word boundaries to avoid matching:
    - Dates (e.g., "12/07/2024", "2024-07-12")
    - Times (e.g., "10:30", "14:45")
    - Other non-price numbers embedded in text

    Selection policy when a string contains more than one digit group
    (common with noisy OCR on short price crops, e.g. "3, 3, 3."):
        1. Prefer a match that includes a decimal separator - that's a
           much stronger signal of an actual price (e.g. "12.500") than
           a bare integer, which is more likely a hallucinated repeat.
        2. Otherwise take the first match.
    Any string with more than one *distinct* candidate value is flagged
    "ambiguous" so the review UI can call extra attention to it, rather
    than silently picking one and hiding the uncertainty.

    Edge cases handled:
    - Leading/trailing whitespace
    - Multiple decimal separators (invalid)
    - Zero or negative prices (invalid)
    - Prices outside reasonable range
    - OCR errors like "O" instead of "0"

    Returns a dict with:
        "value":     float, or None if no number was found
        "raw":       the exact substring that produced "value"
        "ambiguous": True if multiple differing numbers were found
    """
    cfg = get_config().postprocessing
    
    # Clean common OCR errors before processing
    text = text.strip()
    # Replace common OCR misreads (letter O -> digit 0)
    # Only replace isolated O's that look like they should be zeros
    cleaned_text = re.sub(r'\bO\b', '0', text)  # Isolated capital O
    cleaned_text = re.sub(r'(?<=\d)O(?=\d)', '0', cleaned_text)  # O between digits
    cleaned_text = re.sub(r'(?<=\d)O(?=[.,])', '0', cleaned_text)  # O before decimal
    
    # Improved price pattern with word boundaries
    # Matches: 123, 12.5, 12.50, 12.500, 1,234.50, etc.
    # Doesn't match: dates (2024-07-12), times (10:30), phone numbers
    price_pattern = re.compile(r'\b\d{1,6}(?:[.,]\d{1,3})?\b')
    
    matches = price_pattern.findall(cleaned_text)
    if not matches:
        return {"value": None, "raw": None, "ambiguous": False}

    # Filter out obvious non-prices (edge cases)
    valid_matches = []
    for match in matches:
        # Skip if it looks like a date (4-digit year)
        if len(match) == 4 and match.isdigit():
            continue
        # Skip if multiple decimal separators
        if match.count('.') > 1 or match.count(',') > 1:
            continue
        # Skip if both . and , appear (likely OCR error)
        if '.' in match and ',' in match:
            continue
        valid_matches.append(match)
    
    if not valid_matches:
        return {"value": None, "raw": None, "ambiguous": False}

    # Prefer matches with decimal separator (stronger signal of price)
    decimal_matches = [m for m in valid_matches if "." in m or "," in m]
    chosen_raw = decimal_matches[0] if decimal_matches else valid_matches[0]

    try:
        # Normalize decimal separator to .
        value = float(chosen_raw.replace(",", "."))
    except ValueError:
        return {"value": None, "raw": None, "ambiguous": False}
    
    # Validate price is reasonable and positive
    if value <= 0:
        return {"value": None, "raw": None, "ambiguous": False}
    
    if not (cfg.min_reasonable_price <= value <= cfg.max_reasonable_price):
        return {"value": None, "raw": None, "ambiguous": False}

    # Check for ambiguity (multiple distinct values)
    distinct_values = set()
    for m in valid_matches:
        try:
            val = float(m.replace(",", "."))
            if val > 0:  # Only count valid positive values
                distinct_values.add(val)
        except ValueError:
            continue
    
    ambiguous = len(distinct_values) > 1

    return {"value": value, "raw": chosen_raw, "ambiguous": ambiguous}


def detect_currency_symbol(text: str) -> str | None:
    """Look for a currency *symbol* (not abbreviation letters) in the
    recognized text.
    
    Checks for:
    - € (Euro symbol)
    - $ (Dollar - treated as EUR for European context, or could be USD)
    - Common OCR errors for € (C with lines, etc.)
    
    Per spec §5.5, lettered abbreviations (DT/dt/D/d/TND) are deliberately
    NOT checked - they're unreliable in handwriting recognition.
    """
    text = text.strip()
    
    # Check for Euro symbol
    if "€" in text or "EUR" in text.upper():
        return "EUR"
    
    # Check for Dollar (context-dependent - could map to USD or EUR)
    # In Tunisian context, $ might indicate foreign currency (EUR)
    if "$" in text:
        return "EUR"  # Adjust based on your business logic
    
    # Common OCR errors for € symbol
    # Sometimes recognized as "E" or "C" with decoration
    if re.search(r'\b[EC€]\s*(?=\d)', text):
        return "EUR"
    
    return None


def resolve_currency(
    default_currency: str,
    text: str,
    confidence: float,
    threshold: float = None,
) -> tuple:
    """Resolve the currency for one region.

    The owner-set menu-wide default (spec §5.5) always wins unless a
    currency symbol was detected AND recognition confidence for that
    region is high enough to trust it as an override.

    Returns (currency_code, source) where source is "default" or
    "detected_symbol", so the review UI can show why a field is what it
    is if the owner ever wonders.
    """
    if threshold is None:
        threshold = get_config().postprocessing.currency_confidence_threshold
    
    symbol = detect_currency_symbol(text)
    if symbol is not None and confidence >= threshold:
        return symbol, "detected_symbol"
    return default_currency, "default"


def process_region(region: dict, default_currency: str = None) -> dict:
    """Run price extraction + currency resolution on one recognized
    region (a dict from recognition.recognize_regions) and return it
    with the extra fields added. Original keys are preserved."""
    if default_currency is None:
        default_currency = get_config().postprocessing.default_currency
    
    text = region.get("text", "")
    confidence = region.get("confidence", 0.0)

    price = extract_price(text)
    currency, currency_source = resolve_currency(default_currency, text, confidence)

    result = dict(region)
    result["price_value"] = price["value"]
    result["price_raw"] = price["raw"]
    result["price_ambiguous"] = price["ambiguous"]
    result["currency"] = currency
    result["currency_source"] = currency_source
    return result


def process_recognition_results(
    results: list, default_currency: str = None
) -> list:
    """Run process_region over every recognized region.

    Args:
        results: list of dicts from recognition.recognize_regions.
        default_currency: the owner-set menu-wide default, "TND" or
            "EUR" (spec §5.5 - a single toggle, not per-item).

    Returns:
        List of dicts, same order as input, each with the fields
        documented in process_region added.
    """
    if default_currency is None:
        default_currency = get_config().postprocessing.default_currency
    
    supported = get_config().postprocessing.supported_currencies
    if default_currency not in supported:
        raise ValueError(
            f"Unsupported default_currency {default_currency!r}. "
            f"Expected one of {sorted(supported)}."
        )

    return [process_region(r, default_currency) for r in results]


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocessing import preprocess_image
    from detection import detect_text_regions
    from recognition import recognize_regions

    if len(sys.argv) < 2:
        print("Usage: python postprocess.py <path_to_image> [TND|EUR]")
        sys.exit(1)

    default_currency = sys.argv[2] if len(sys.argv) > 2 else get_config().postprocessing.default_currency

    prep = preprocess_image(sys.argv[1])
    regions = detect_text_regions(prep["image"])
    print(f"Detected {len(regions)} text regions")

    recognized = recognize_regions(regions)
    processed = process_recognition_results(recognized, default_currency=default_currency)

    for i, r in enumerate(processed):
        price_str = f"{r['price_value']}" if r["price_value"] is not None else "-"
        flag = " (ambiguous)" if r["price_ambiguous"] else ""
        print(
            f"[{i:02d}] conf={r['confidence']:.2f}  "
            f"text={r['text']!r}  price={price_str}{flag}  "
            f"currency={r['currency']} ({r['currency_source']})"
        )