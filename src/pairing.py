"""
Advanced item-price pairing using tilted horizontal line algorithm.

This module implements the pairing algorithm from Experiment/pairing_experiment.py,
which outperforms the simple y-distance nearest-neighbor approach.

Algorithm:
1. For each item box, draw a horizontal line through its right-center point
2. Find boxes to the right that the line collides with
3. If no collision, tilt the line ±1°, ±2°, etc. up to MAX_TILT_DEG
4. Two-pass system:
   - Pass 1: Match only unambiguous pairs (clear winner by tilt angle)
   - Pass 2: Re-verify remaining orphans using learned vertical offset from pass 1
5. Classify category headers vs items based on height
6. Group items under nearest category header above

Why this works better than simple y-distance:
- Handles row alignment issues where items/prices aren't perfectly horizontal
- Two-pass approach avoids cascade errors from ambiguous matches
- Position-based (not text-based), so broken OCR doesn't break pairing
"""

import math
from typing import Optional, List, Tuple, Dict, Set
from logging_config import get_logger

logger = get_logger(__name__)

# Tunables from pairing_experiment.py
MAX_TILT_DEG = 8.0              # Max angle to tilt pairing line
CATEGORY_HEIGHT_RATIO = 1.35    # Box taller than median*this = category
NOISE_HEIGHT_RATIO = 4.0        # Box taller than median*this = decorative
                                 # illustration blob, not text - excluded
                                 # entirely from classification/pairing AND
                                 # from recognition (see build_skeleton).
                                 # This was present in pairing_experiment.py
                                 # but got dropped when the algorithm was
                                 # ported here - restored because leaving
                                 # illustration blobs in the height pool
                                 # skews the median that category_cutoff is
                                 # computed from, which was misclassifying
                                 # real categories/items around them.
MAX_VERTICAL_SHIFT_RATIO = 1.0  # Max row-jumping (multiples of median height)
AMBIGUITY_MARGIN_DEG = 1.5      # Degree difference to treat as tie


class Box:
    """Represents a text box with pairing metadata."""
    
    def __init__(self, idx: int, region: dict):
        self.idx = idx
        self.region = region  # Original region from detection
        self.kind = "unknown"  # "category" | "item" | "unknown"
        self.paired_with: Optional[int] = None
        self.pair_tilt_deg: Optional[float] = None
        self.verified_pass = False  # True if resolved by pass-2
        self.group: Optional[int] = None  # Index of category header
        
        # Extract coordinates from box corners if available, otherwise from x,y centers
        if "box" in region:
            box = region["box"]
            self.xmin = float(box[:, 0].min())
            self.xmax = float(box[:, 0].max())
            self.ymin = float(box[:, 1].min())
            self.ymax = float(box[:, 1].max())
            self.width = self.xmax - self.xmin
            self.height = self.ymax - self.ymin
            self.xcenter = (self.xmin + self.xmax) / 2
            self.ycenter = (self.ymin + self.ymax) / 2
        else:
            # Fallback if no box field (shouldn't happen but be defensive)
            self.xcenter = region.get("x", 0)
            self.ycenter = region.get("y", 0)
            # Estimate dimensions (crude fallback)
            self.width = 100  
            self.height = 20
            self.xmin = self.xcenter - self.width / 2
            self.xmax = self.xcenter + self.width / 2
            self.ymin = self.ycenter - self.height / 2
            self.ymax = self.ycenter + self.height / 2


def classify_boxes(boxes: List[Box]) -> float:
    """
    Classify boxes as 'noise' (decorative/illustration, excluded entirely),
    'category' (headers), or 'item' based on height.

    Returns median item height for use in pairing.
    """
    if not boxes:
        return 0.0
    
    heights = sorted(b.height for b in boxes)
    n = len(heights)
    # Trim top 20% to avoid large illustration blobs affecting median
    trimmed = heights[:max(1, int(n * 0.8))]
    median_h = trimmed[len(trimmed) // 2]
    
    category_cutoff = median_h * CATEGORY_HEIGHT_RATIO
    noise_cutoff = median_h * NOISE_HEIGHT_RATIO

    for b in boxes:
        if b.height > noise_cutoff:
            b.kind = "noise"
        elif b.height > category_cutoff:
            b.kind = "category"
        else:
            b.kind = "item"

    logger.debug(f"Classified {sum(1 for b in boxes if b.kind == 'noise')} noise, "
                 f"{sum(1 for b in boxes if b.kind == 'category')} categories, "
                 f"{sum(1 for b in boxes if b.kind == 'item')} items")
    
    return median_h


def line_y_at_x(y0: float, x0: float, x: float, angle_deg: float) -> float:
    """Calculate y-coordinate of tilted line at given x."""
    return y0 + (x - x0) * math.tan(math.radians(angle_deg))


def find_pair_candidates(
    source: Box,
    candidates: List[Box],
    used: Set[int],
    max_vertical_shift: float
) -> List[Tuple[float, float, Box, float]]:
    """
    Find and rank all valid pairing candidates for a source box.
    
    Returns list of (abs_angle, x_distance, candidate, signed_angle)
    sorted by abs_angle (smallest first).
    
    This solves for the exact angle needed to reach each candidate,
    avoiding the ordering bug of trying preset angles (0, +1, -1, +2, -2...).
    """
    x0, y0 = source.xmax, source.ycenter
    ranked = []
    
    for cand in candidates:
        if cand.idx == source.idx or cand.idx in used:
            continue
        if cand.kind != "item":
            continue
        if cand.xmin <= source.xmax:
            continue  # Must be strictly to the right
        
        dx = cand.xcenter - x0
        dy = cand.ycenter - y0
        
        if abs(dy) > max_vertical_shift:
            continue  # Too far vertically
        
        angle = math.degrees(math.atan2(dy, dx))
        if abs(angle) > MAX_TILT_DEG:
            continue
        
        ranked.append((abs(angle), cand.xmin - source.xmax, cand, round(angle, 1)))
    
    ranked.sort(key=lambda t: (t[0], t[1]))  # Sort by abs_angle, then x_distance
    return ranked


def estimate_row_pitch(boxes: List[Box]) -> float:
    """Estimate median vertical spacing between item rows."""
    ys = sorted(b.ycenter for b in boxes if b.kind == "item")
    diffs = [b - a for a, b in zip(ys, ys[1:]) if b - a > 2]
    if not diffs:
        return 30.0
    diffs.sort()
    return diffs[len(diffs) // 2]


def commit_pair(a: Box, b: Box, angle: float, used: Set[int], verified: bool = False) -> None:
    """Mark two boxes as paired."""
    a.paired_with = b.idx
    a.pair_tilt_deg = angle
    a.verified_pass = verified
    b.paired_with = a.idx
    b.pair_tilt_deg = angle
    b.verified_pass = verified
    used.add(a.idx)
    used.add(b.idx)


def run_pairing(boxes: List[Box], median_item_height: float) -> None:
    """
    Two-pass pairing algorithm.
    
    Pass 1: Match only unambiguous pairs (clear winner by angle)
    Pass 2: Match remaining orphans using learned vertical offset
    """
    row_pitch = estimate_row_pitch(boxes)
    max_vertical_shift = max(
        median_item_height * MAX_VERTICAL_SHIFT_RATIO,
        row_pitch * 0.6
    )
    
    items = [b for b in boxes if b.kind == "item"]
    used: Set[int] = set()
    confident_dys = []
    
    # Pass 1: Unambiguous matches only
    for src in items:
        if src.idx in used:
            continue
        
        ranked = find_pair_candidates(src, items, used, max_vertical_shift)
        if not ranked:
            continue
        
        best_abs_angle, _, best_cand, best_angle = ranked[0]
        
        # Check for ambiguity
        if len(ranked) > 1:
            second_abs_angle = ranked[1][0]
            if (second_abs_angle - best_abs_angle) < AMBIGUITY_MARGIN_DEG:
                continue  # Too close to call - defer to pass 2
        
        commit_pair(src, best_cand, best_angle, used)
        confident_dys.append(best_cand.ycenter - src.ycenter)
    
    logger.debug(f"Pass 1: Paired {len(used) // 2} unambiguous matches")
    
    # Learn typical vertical offset from confident matches
    if confident_dys:
        confident_dys.sort()
        learned_dy = confident_dys[len(confident_dys) // 2]
    else:
        learned_dy = 0.0
    
    # Pass 2: Re-verify remaining items using learned offset
    remaining = [b for b in items if b.idx not in used]
    for src in remaining:
        if src.idx in used:
            continue
        
        expected_y = src.ycenter + learned_dy
        best, best_dist = None, None
        
        for cand in remaining:
            if cand.idx == src.idx or cand.idx in used:
                continue
            if cand.xmin <= src.xmax:
                continue
            
            dist = abs(cand.ycenter - expected_y)
            if dist > max_vertical_shift:
                continue
            
            if best_dist is None or dist < best_dist:
                best, best_dist = cand, dist
        
        if best is not None:
            angle = round(
                math.degrees(math.atan2(
                    best.ycenter - src.ycenter,
                    best.xcenter - src.xcenter
                )),
                1
            )
            commit_pair(src, best, angle, used, verified=True)
    
    logger.debug(f"Pass 2: Paired {(len(used) - len(confident_dys) * 2) // 2} additional matches")


def assign_groups(boxes: List[Box]) -> List[Box]:
    """Group item boxes under nearest category header above."""
    categories = sorted(
        [b for b in boxes if b.kind == "category"],
        key=lambda b: b.ycenter
    )
    
    for b in boxes:
        if b.kind != "item":
            continue
        
        best_cat = None
        for cat in categories:
            if cat.ycenter < b.ycenter:
                best_cat = cat
            else:
                break
        
        b.group = best_cat.idx if best_cat is not None else None
    
    return categories


def build_skeleton(regions: List[dict]) -> List[dict]:
    """
    Build the category/item/pairing skeleton straight from detection
    output - BEFORE recognition ever runs.

    Every decision made here (noise vs. category vs. item, which name
    pairs with which price, which category a box belongs under) is
    computed purely from box geometry (position, height, angle). None of
    it depends on recognized text, so none of it can be corrupted by a
    bad OCR read - a box's structural role in the menu is settled before
    its content is ever known. This also means noise boxes (illustrations,
    decorative headers) are identified and can be dropped BEFORE the
    (expensive) recognition step ever sees them, instead of after.

    Args:
        regions: list of dicts from detection.detect_text_regions - only
            "box", "x", "y" are used; "crop"/"text" are not needed here.

    Returns:
        List of dicts, same length/order as `regions`, each with:
            "box_id":   index into `regions` - use this to re-attach
                        recognized text/price to the right box later
            "role":     "category" | "item_name" | "item_price" |
                        "unresolved" | "noise"
                "unresolved" = an item-kind box that never found a
                pairing partner by geometry. This is the one case
                geometry genuinely can't settle on its own: a price
                written with no item nearby and an item with no price
                (e.g. "market price") look identical from position
                alone. It's left for the assembly stage to resolve
                using the same price-parsing logic as everything else,
                once recognized text is available.
            "pair_id":  box_id of its paired counterpart, or None
            "group_id": box_id of the category header it sits under
                        (item-kind boxes only), or None
    """
    if not regions:
        return []

    boxes = [Box(i, r) for i, r in enumerate(regions)]
    median_height = classify_boxes(boxes)
    run_pairing(boxes, median_height)
    assign_groups(boxes)

    skeleton = []
    for box in boxes:
        if box.kind == "noise":
            role, pair_id = "noise", None
        elif box.kind == "category":
            role, pair_id = "category", None
        elif box.paired_with is not None:
            other = boxes[box.paired_with]
            role = "item_name" if box.xcenter < other.xcenter else "item_price"
            pair_id = other.idx
        else:
            role, pair_id = "unresolved", None

        skeleton.append({
            "box_id": box.idx,
            "role": role,
            "pair_id": pair_id,
            "group_id": box.group if box.kind == "item" else None,
        })

    logger.info(
        "Skeleton built",
        extra={
            "categories": sum(1 for s in skeleton if s["role"] == "category"),
            "paired_item_boxes": sum(1 for s in skeleton if s["role"] in ("item_name", "item_price")),
            "unresolved": sum(1 for s in skeleton if s["role"] == "unresolved"),
            "noise_excluded": sum(1 for s in skeleton if s["role"] == "noise"),
        },
    )
    return skeleton


def build_skeleton(regions: List[dict]) -> List[dict]:
    """
    Build the category/item/pairing skeleton straight from detection
    output - BEFORE recognition ever runs.

    Every decision made here (noise vs. category vs. item, which name
    pairs with which price, which category a box belongs under) is
    computed purely from box geometry (position, height, angle). None of
    it depends on recognized text, so none of it can be corrupted by a
    bad OCR read - a box's structural role in the menu is settled before
    its content is ever known. This also means noise boxes (illustrations,
    decorative headers) are identified and can be dropped BEFORE the
    (expensive) recognition step ever sees them, instead of after.

    Args:
        regions: list of dicts from detection.detect_text_regions - only
            "box", "x", "y" are used; "crop"/"text" are not needed here.

    Returns:
        List of dicts, same length/order as `regions`, each with:
            "box_id":   index into `regions` - use this to re-attach
                        recognized text/price to the right box later
            "role":     "category" | "item_name" | "item_price" |
                        "unresolved" | "noise"
                "unresolved" = an item-kind box that never found a
                pairing partner by geometry. This is the one case
                geometry genuinely can't settle on its own: a price
                written with no item nearby and an item with no price
                (e.g. "market price") look identical from position
                alone. It's left for the assembly stage to resolve
                using the same price-parsing logic as everything else,
                once recognized text is available.
            "pair_id":  box_id of its paired counterpart, or None
            "group_id": box_id of the category header it sits under
                        (item-kind boxes only), or None
    """
    if not regions:
        return []

    boxes = [Box(i, r) for i, r in enumerate(regions)]
    median_height = classify_boxes(boxes)
    run_pairing(boxes, median_height)
    assign_groups(boxes)

    skeleton = []
    for box in boxes:
        if box.kind == "noise":
            role, pair_id = "noise", None
        elif box.kind == "category":
            role, pair_id = "category", None
        elif box.paired_with is not None:
            other = boxes[box.paired_with]
            role = "item_name" if box.xcenter < other.xcenter else "item_price"
            pair_id = other.idx
        else:
            role, pair_id = "unresolved", None

        skeleton.append({
            "box_id": box.idx,
            "role": role,
            "pair_id": pair_id,
            "group_id": box.group if box.kind == "item" else None,
        })

    logger.info(
        "Skeleton built",
        extra={
            "categories": sum(1 for s in skeleton if s["role"] == "category"),
            "paired_item_boxes": sum(1 for s in skeleton if s["role"] in ("item_name", "item_price")),
            "unresolved": sum(1 for s in skeleton if s["role"] == "unresolved"),
            "noise_excluded": sum(1 for s in skeleton if s["role"] == "noise"),
        },
    )
    return skeleton


def pair_regions_advanced(regions: List[dict]) -> Tuple[List[dict], List[dict]]:
    """
    LEGACY: pairs item names with prices AFTER recognition, using already-
    recognized regions. Superseded by build_skeleton(), which runs the
    same underlying geometry (classify_boxes/run_pairing/assign_groups)
    right after detection, before recognition - kept here only for any
    caller still depending on this post-recognition entry point.

    Pair item names with prices using advanced tilted-line algorithm.
    
    This replaces the simple y-distance pairing in pipeline.py with the
    more robust algorithm from pairing_experiment.py.
    
    Args:
        regions: List of text regions from postprocess (with text, confidence, price_value, etc.)
    
    Returns:
        (items, orphan_prices) tuple:
            - items: List of dicts with name, price, currency, is_category_header
            - orphan_prices: List of price regions that couldn't be paired
    """
    if not regions:
        return [], []
    
    # Create Box objects for pairing algorithm
    boxes = [Box(i, r) for i, r in enumerate(regions)]
    
    # Classify categories vs items
    median_height = classify_boxes(boxes)
    
    # Run two-pass pairing
    run_pairing(boxes, median_height)
    
    # Assign category groups
    categories = assign_groups(boxes)
    
    # Convert back to item dicts
    items = []
    orphan_prices = []
    paired_indices = set()
    
    for box in sorted(boxes, key=lambda b: b.ycenter):
        region = box.region
        
        if box.kind == "category":
            # Category header
            items.append({
                "name": region["text"],
                "name_confidence": region["confidence"],
                "price_value": None,
                "price_raw": None,
                "price_ambiguous": False,
                "price_confidence": None,
                "currency": None,
                "currency_source": None,
                "is_category_header": True,
            })
        elif box.kind == "item":
            if box.paired_with is not None and box.idx not in paired_indices:
                # Paired item
                other_box = boxes[box.paired_with]
                other_region = other_box.region
                
                # Determine which is name and which is price
                # (assume left is name, right is price)
                if box.xcenter < other_box.xcenter:
                    name_region, price_region = region, other_region
                else:
                    name_region, price_region = other_region, region
                
                items.append({
                    "name": name_region["text"],
                    "name_confidence": name_region["confidence"],
                    "price_value": price_region.get("price_value"),
                    "price_raw": price_region.get("price_raw"),
                    "price_ambiguous": price_region.get("price_ambiguous", False),
                    "price_confidence": price_region["confidence"],
                    "currency": price_region.get("currency"),
                    "currency_source": price_region.get("currency_source"),
                    "is_category_header": False,
                })
                
                # Mark both boxes as processed
                paired_indices.add(box.idx)
                paired_indices.add(other_box.idx)
            
            elif box.paired_with is None:
                # Orphan - could be item without price or unpaired price
                has_price = region.get("price_value") is not None
                
                if has_price:
                    # Orphan price
                    orphan_prices.append({
                        "text": region["text"],
                        "confidence": region["confidence"],
                        "price_value": region.get("price_value"),
                        "price_raw": region.get("price_raw"),
                    })
                else:
                    # Item without price (treat as category header)
                    items.append({
                        "name": region["text"],
                        "name_confidence": region["confidence"],
                        "price_value": None,
                        "price_raw": None,
                        "price_ambiguous": False,
                        "price_confidence": None,
                        "currency": None,
                        "currency_source": None,
                        "is_category_header": True,
                    })
    
    logger.info(f"Pairing complete: {len(items)} items, {len(orphan_prices)} orphan prices")
    
    return items, orphan_prices