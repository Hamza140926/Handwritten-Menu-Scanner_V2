"""
Category/item/price pairing — geometry-only, runs right after detection,
before recognition ever sees the boxes.

This is the merged version: production shell (dict-based Box construction,
build_skeleton, assign_groups, logging) from the original pairing.py,
combined with the classification/pairing ENGINE from
Experiment/experiment2.py, which outperforms the old height-ratio +
two-pass-tilted-line approach:

  - classify_boxes: noise flagged by SHAPE (area + aspect), not height.
    category vs item is NOT a height cutoff - it's a 3-signal vote
    (price-column absence / row isolation / large gap-above), because a
    fixed "taller than median*X" threshold breaks constantly on
    handwriting where scale drifts across the page.

  - run_pairing: name<->price pairing via the HUNGARIAN ALGORITHM
    (scipy.optimize.linear_sum_assignment) instead of a two-pass
    tilted-line nearest-neighbor search. This finds the globally optimal
    set of pairs instead of a first-come-first-served match. A
    cost-threshold rejection (dummy padding) leaves a box "unresolved"
    rather than force-pairing it to the least-bad option.

Known, honest limitation (unchanged from before): an item with a
genuinely missing price and a category header look identical from
geometry alone (both are "alone on their row"). That box lands as
"category". Resolving that needs recognized text - out of scope here,
left for assembly to sort out from the "unresolved" role once text is
available.
"""

import math
from typing import Optional, List, Tuple, Set

import numpy as np
from scipy.optimize import linear_sum_assignment

from logging_config import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------
# Tunables (from experiment2.py - the voting/Hungarian engine)
# --------------------------------------------------------------------------
NOISE_AREA_RATIO = 0.05     # box area > this fraction of image area...
NOISE_ASPECT_MIN = 2.0      # ...AND width/height below this -> noise blob
                             # (illustrations are big AND squarish/tall;
                             # text lines are consistently wide, so both
                             # conditions together are needed, not just size)

COLUMN_X_TOLERANCE = 15     # px - how close two boxes' left edges must be
                             # to count as "same column"
MIN_COLUMN_COLLISIONS_FOR_ITEM = 2  # min boxes sharing an x-position to
                                     # count it as an "established" column

GAP_RATIO = 1.6              # gap above a row > row_pitch * this -> extra
                              # whitespace vote for "new section starts here"
CATEGORY_VOTE_THRESHOLD = 2  # need at least this many of the 3 votes below

MAX_TILT_DEG = 8.0                 # max angle allowed between a
                                    # name/price candidate pair
MAX_VERTICAL_SHIFT_RATIO = 1.2     # max vertical offset, as a multiple of
                                    # median item height, before a
                                    # candidate is rejected outright

REJECT_COST = 999.0                # cost assigned to "no match" dummy
                                    # nodes in the Hungarian cost matrix;
                                    # any real pair costing >= this is
                                    # treated as no-match too
W_ANGLE = 1.0                      # cost weights
W_VDIST = 0.08
BIG = 1e6                          # cost for structurally-invalid pairs
                                    # (wrong direction / too far), keeps
                                    # them out of the solution entirely


class Box:
    """Represents a text box with classification + pairing metadata."""

    def __init__(self, idx: int, region: dict):
        self.idx = idx
        self.region = region  # Original region from detection
        self.kind = "unknown"  # "noise" | "category" | "item"

        # filled in by classify_boxes()
        self.column_collisions = 0

        # filled in by pairing
        self.paired_with: Optional[int] = None
        self.pair_cost: Optional[float] = None
        self.role: Optional[str] = None  # "item_name" | "item_price"

        # filled in by grouping
        self.group: Optional[int] = None  # box_id of category header

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


# --------------------------------------------------------------------------
# Classification: noise by shape, category/item by a 3-signal vote
# --------------------------------------------------------------------------
def is_noise(box: Box, image_area: float) -> bool:
    """Flag decorative illustration blobs by SHAPE, not height:
    illustrations tend to be large AND roughly as tall as they are
    wide, unlike text lines which are consistently much wider than
    tall. Both conditions are required together - either alone is too
    trigger-happy (a genuinely huge banner headline is wide, not
    square; a small logo isn't large enough to matter)."""
    area_ratio = (box.width * box.height) / image_area
    aspect = box.width / max(box.height, 1.0)
    return area_ratio >= NOISE_AREA_RATIO and aspect < NOISE_ASPECT_MIN


def find_column_lines(boxes: List[Box]) -> Tuple[Optional[float], Optional[float]]:
    """Find the page's dominant vertical column x-positions by binning
    every box's xmin (bin width = COLUMN_X_TOLERANCE) and keeping bins
    with enough members to count as a real, repeated column.

    Returns (name_column_x, price_column_x) - the leftmost and
    rightmost established columns - or (None, None) if the page
    doesn't show a clean multi-column structure, in which case the
    price-column vote is skipped for every box.
    """
    if len(boxes) < 4:
        return None, None

    xmins = sorted(b.xmin for b in boxes)
    bins: List[List[float]] = []
    for x in xmins:
        if bins and (x - bins[-1][-1]) <= COLUMN_X_TOLERANCE:
            bins[-1].append(x)
        else:
            bins.append([x])

    established = [b for b in bins if len(b) >= MIN_COLUMN_COLLISIONS_FOR_ITEM]
    if len(established) < 2:
        return None, None

    centers = sorted(sum(b) / len(b) for b in established)
    return centers[0], centers[-1]


def has_row_companion_at(boxes: List[Box], box: Box, column_x: float) -> bool:
    """Does some OTHER box sit near `column_x` and vertically overlap
    `box`'s row? Cast the column's vertical line down the page and see
    if it passes through anything on this specific row."""
    for o in boxes:
        if o.idx == box.idx:
            continue
        if abs(o.xmin - column_x) > COLUMN_X_TOLERANCE * 1.5:
            continue
        if o.ymin <= box.ymax and o.ymax >= box.ymin:  # row overlap
            return True
    return False


def is_alone_on_row(box: Box, boxes: List[Box]) -> bool:
    """True if nothing else on the page vertically overlaps this box's
    row, in ANY column. A more general version of the price-column
    check above - also works when there's no distinct price column at
    all (e.g. a single-column list where prices trail inline)."""
    for o in boxes:
        if o.idx == box.idx:
            continue
        if o.ymin <= box.ymax and o.ymax >= box.ymin:
            return False
    return True


def estimate_row_pitch(boxes: List[Box]) -> float:
    """Median vertical spacing between consecutive rows on the page -
    used only as a *relative* reference for the gap-above vote, never
    as a hard cutoff on its own."""
    ys = sorted(b.ycenter for b in boxes)
    diffs = [y2 - y1 for y1, y2 in zip(ys, ys[1:]) if y2 - y1 > 2]
    if not diffs:
        return 30.0
    diffs.sort()
    return diffs[len(diffs) // 2]


def gap_above(box: Box, boxes: List[Box]) -> Optional[float]:
    """Vertical gap between this box and the nearest box fully above
    it. Returns None if nothing is above (top of page) - excluded from
    the vote rather than guessed at."""
    above = [o for o in boxes if o.idx != box.idx and o.ymax <= box.ymin + 2]
    if not above:
        return None
    nearest = max(above, key=lambda o: o.ymax)
    return box.ymin - nearest.ymax


def classify_boxes(boxes: List[Box], image_area: float) -> float:
    """Classify boxes as noise / category / item. Returns the median
    item height, kept only as a distance reference for pairing later -
    it plays no role in the category/item decision.

    NOTE the signature change from the old height-ratio classifier:
    this one requires image_area (used by is_noise's shape check).
    Callers must pass the source image's height*width.
    """
    if not boxes:
        return 0.0

    for b in boxes:
        b.kind = "noise" if is_noise(b, image_area) else "unknown"

    non_noise = [b for b in boxes if b.kind != "noise"]
    if not non_noise:
        return 0.0

    heights = sorted(b.height for b in non_noise)
    n = len(heights)
    median_h = heights[n // 2]

    row_pitch = estimate_row_pitch(non_noise)
    name_col_x, price_col_x = find_column_lines(non_noise)

    for b in non_noise:
        b.column_collisions = sum(
            1 for o in non_noise
            if o.idx != b.idx and abs(o.xmin - b.xmin) <= COLUMN_X_TOLERANCE
        )

        votes = 0

        # Vote (a): no companion in the established price column on this row
        if price_col_x is not None:
            has_price_companion = (
                abs(b.xmin - price_col_x) <= COLUMN_X_TOLERANCE * 1.5
                or has_row_companion_at(non_noise, b, price_col_x)
            )
            if not has_price_companion:
                votes += 1

        # Vote (b): nothing at all overlaps this row, in any column
        if is_alone_on_row(b, non_noise):
            votes += 1

        # Vote (c): unusually large whitespace gap above this row
        gap = gap_above(b, non_noise)
        if gap is not None and gap > row_pitch * GAP_RATIO:
            votes += 1

        b.kind = "category" if votes >= CATEGORY_VOTE_THRESHOLD else "item"

    logger.debug(
        "Classified boxes",
        extra={
            "noise": sum(1 for b in boxes if b.kind == "noise"),
            "category": sum(1 for b in boxes if b.kind == "category"),
            "item": sum(1 for b in boxes if b.kind == "item"),
        },
    )

    return median_h


# --------------------------------------------------------------------------
# Pairing: Hungarian algorithm with reject-threshold padding
# --------------------------------------------------------------------------
def build_cost_matrix(items: List[Box], median_h: float) -> np.ndarray:
    """NxN cost matrix. cost[i][j] = cost of treating items[i] as the NAME
    and items[j] as the PRICE of the same line. Structurally invalid pairs
    (wrong direction, too far, too tilted) get cost BIG so Hungarian will
    never choose them over a dummy no-match slot."""
    n = len(items)
    cost = np.full((n, n), BIG, dtype="float64")
    max_vshift = max(median_h * MAX_VERTICAL_SHIFT_RATIO, 1.0)

    for i, src in enumerate(items):
        for j, cand in enumerate(items):
            if i == j:
                continue
            if cand.xmin <= src.xmax:
                continue  # must be strictly to the right

            dx = cand.xcenter - src.xmax
            dy = cand.ycenter - src.ycenter
            if abs(dy) > max_vshift:
                continue

            angle = math.degrees(math.atan2(dy, dx))
            if abs(angle) > MAX_TILT_DEG:
                continue

            cost[i, j] = W_ANGLE * abs(angle) + W_VDIST * abs(dy)

    return cost


def pad_with_rejection(cost: np.ndarray) -> np.ndarray:
    """Pad an NxN cost matrix to 2Nx2N so Hungarian has a 'no match'
    option for every row and column, instead of being forced into a
    complete matching. Standard assignment-with-rejection trick:

        [ real         | reject_rows ]
        [ reject_cols  | zero        ]
    """
    n = cost.shape[0]
    top = np.hstack([cost, np.full((n, n), REJECT_COST)])
    bottom = np.hstack([np.full((n, n), REJECT_COST), np.zeros((n, n))])
    return np.vstack([top, bottom])


def run_pairing(boxes: List[Box], median_item_height: float) -> None:
    """Pair item-kind boxes using the Hungarian algorithm on the padded
    cost matrix, then resolve any box used in two accepted pairs
    (possible because the bipartite relaxation lets the same box appear
    as both a 'name' row and a 'price' column) by keeping the cheaper
    one and dropping the other back to unresolved.

    NOTE: same public name/signature (boxes, median_item_height) as the
    old two-pass tilted-line version, so callers don't need to change -
    only classify_boxes()'s signature changed (needs image_area).
    """
    items = [b for b in boxes if b.kind == "item"]
    if len(items) < 2:
        return

    real_cost = build_cost_matrix(items, median_item_height)
    padded = pad_with_rejection(real_cost)
    row_ind, col_ind = linear_sum_assignment(padded)

    n = len(items)
    accepted: List[Tuple[float, int, int]] = []  # (cost, name_idx, price_idx)
    for r, c in zip(row_ind, col_ind):
        if r < n and c < n:  # both sides are real boxes, not dummies
            c_val = real_cost[r, c]
            if c_val < REJECT_COST:
                accepted.append((c_val, r, c))

    # Resolve double-use: sort by cost (best matches first, thanks to
    # Hungarian's global optimization) and greedily commit pairs where
    # neither box has been claimed yet.
    accepted.sort(key=lambda t: t[0])
    used: Set[int] = set()
    for c_val, i, j in accepted:
        name_box, price_box = items[i], items[j]
        if name_box.idx in used or price_box.idx in used:
            continue
        name_box.paired_with = price_box.idx
        name_box.role = "item_name"
        name_box.pair_cost = c_val
        price_box.paired_with = name_box.idx
        price_box.role = "item_price"
        price_box.pair_cost = c_val
        used.add(name_box.idx)
        used.add(price_box.idx)

    logger.debug("Pairing complete", extra={"pairs": len(used) // 2, "items": n})


# --------------------------------------------------------------------------
# Grouping: attach items to the nearest category header above them
# --------------------------------------------------------------------------
def assign_groups(boxes: List[Box]) -> List[Box]:
    """Group item boxes under nearest category header above. Returns the
    sorted list of category boxes (kept for callers that want it, e.g.
    interactive_detection's stats)."""
    categories = sorted(
        [b for b in boxes if b.kind == "category"],
        key=lambda b: b.ycenter,
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


# --------------------------------------------------------------------------
# Skeleton builder (unchanged shape - production entry point)
# --------------------------------------------------------------------------
def build_skeleton(regions: List[dict], image_area: float) -> List[dict]:
    """
    Build the category/item/pairing skeleton straight from detection
    output - BEFORE recognition ever runs.

    Every decision made here (noise vs. category vs. item, which name
    pairs with which price, which category a box belongs under) is
    computed purely from box geometry (position, size, shape, angle).
    None of it depends on recognized text, so none of it can be
    corrupted by a bad OCR read - a box's structural role in the menu
    is settled before its content is ever known. This also means noise
    boxes (illustrations, decorative headers) are identified and can be
    dropped BEFORE the (expensive) recognition step ever sees them,
    instead of after.

    Args:
        regions: list of dicts from detection.detect_text_regions - only
            "box", "x", "y" are used; "crop"/"text" are not needed here.
        image_area: source image height*width (float). Required by the
            noise-shape check (is_noise) - pass
            float(image.shape[0] * image.shape[1]) from whatever BGR
            array was fed into detection.

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
    median_height = classify_boxes(boxes, image_area)
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