"""
experiment2.py

Standalone experiment for messy handwritten-menu layouts.

Pipeline:
  1. Detect text boxes on the image (PaddleOCR TextDetection).
  2. Classify each box as noise / category / item.
       - noise: flagged by SHAPE (large area + roughly square/tall
         aspect ratio), not height alone - illustrations are big AND
         blob-shaped, unlike text lines which are consistently wide.
       - category vs item: NOT decided by a height-ratio cutoff. A
         fixed "taller than median*X" threshold breaks constantly on
         handwriting, where scale drifts across the page and headers
         often aren't drawn meaningfully bigger than item text at all.
         Instead, three independent, non-height votes are combined:
           (a) no companion sitting in the established price column on
               this row
           (b) nothing at all overlaps this row, in any column
               (row isolation - works even with no distinct price
               column, e.g. single-column layouts)
           (c) unusually large whitespace gap above this row, relative
               to the page's typical row-to-row spacing
         A box needs at least 2 of these 3 votes to be called a
         category. No single fragile threshold decides anything alone.
  3. Pair item-name boxes with item-price boxes using the HUNGARIAN
     ALGORITHM (scipy.optimize.linear_sum_assignment) instead of a
     greedy nearest-candidate search. This finds the globally optimal
     set of pairs instead of a first-come-first-served match, which is
     what breaks on messy/crowded rows. A cost-threshold rejection
     (dummy padding) means a box is left "unresolved" rather than
     force-paired to the least-bad option when nothing decent exists.
     A post-pass resolves the case where the bipartite relaxation
     assigns the same box to two different roles, using a "used-once"
     greedy pass ranked by Hungarian's globally-optimal costs.
  4. Group item boxes under the nearest category header above them.
  5. Draw a debug image and print a terminal report.

Usage:
    python experiment2.py path/to/menu.jpg [--out debug.png]

Known, honest limitation: an item with a genuinely missing price and a
category header still look identical from geometry alone (both are
"alone on their row"). That box will land as "category". Resolving
that case needs recognized text (a short closed-vocabulary header word
vs. no price pattern nearby) - out of scope for this geometry-only pass.

Note: PaddleOCR downloads its detector weights on first run (needs
network access to its model host). Everything past detection
(classification / pairing / drawing / report) has no such dependency
and can be exercised directly by feeding boxes from any detector.
"""

import argparse
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------------------
# Tunables
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


# --------------------------------------------------------------------------
# Box
# --------------------------------------------------------------------------
class Box:
    """A detected text region plus derived geometry + classification state."""

    def __init__(self, idx: int, poly: np.ndarray):
        poly = np.array(poly, dtype="float32")
        self.idx = idx
        self.poly = poly
        self.xmin = float(poly[:, 0].min())
        self.xmax = float(poly[:, 0].max())
        self.ymin = float(poly[:, 1].min())
        self.ymax = float(poly[:, 1].max())
        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin
        self.xcenter = (self.xmin + self.xmax) / 2
        self.ycenter = (self.ymin + self.ymax) / 2

        # filled in by classify_boxes()
        self.kind = "unknown"           # "noise" | "category" | "item"
        self.column_collisions = 0

        # filled in by pairing
        self.paired_with: Optional[int] = None
        self.pair_cost: Optional[float] = None
        self.role: Optional[str] = None  # "item_name" | "item_price"

        # filled in by grouping
        self.group: Optional[int] = None  # box_id of category header


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def detect_boxes(image: np.ndarray) -> List[Box]:
    """Run PaddleOCR's text detector and wrap results as Box objects."""
    from paddleocr import TextDetection

    # enable_mkldnn=False works around a known bug in PaddlePaddle 3.3.x's
    # CPU inference backend (oneDNN/PIR executor) that throws
    # "NotImplementedError: ConvertPirAttribute2RuntimeAttribute not
    # support [...]" on CPU inference with MKL-DNN enabled (the default).
    # See: github.com/PaddlePaddle/Paddle/issues/77340
    detector = TextDetection(model_name="PP-OCRv5_mobile_det", enable_mkldnn=False)
    output = detector.predict(input=image, batch_size=1)
    result_item = next(iter(output))

    polys = _extract_polygons(result_item)
    return [Box(i, p) for i, p in enumerate(polys)]


def _extract_polygons(result_item) -> list:
    """Defensive extraction of dt_polys across PaddleOCR 3.x result shapes."""
    for getter in (
        lambda r: r["dt_polys"],
        lambda r: r["res"]["dt_polys"],
        lambda r: r.json["res"]["dt_polys"],
    ):
        try:
            return list(getter(result_item))
        except (KeyError, TypeError, AttributeError):
            continue
    raise RuntimeError(
        f"Could not find 'dt_polys' in PaddleOCR result. Raw result: {result_item}"
    )


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
    it plays no role in the category/item decision."""
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


def run_pairing_hungarian(boxes: List[Box], median_h: float) -> None:
    """Pair item-kind boxes using the Hungarian algorithm on the padded
    cost matrix, then resolve any box used in two accepted pairs
    (possible because the bipartite relaxation lets the same box appear
    as both a 'name' row and a 'price' column) by keeping the cheaper
    one and dropping the other back to unresolved."""
    items = [b for b in boxes if b.kind == "item"]
    if len(items) < 2:
        return

    real_cost = build_cost_matrix(items, median_h)
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
    used = set()
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


# --------------------------------------------------------------------------
# Grouping: attach items to the nearest category header above them
# --------------------------------------------------------------------------
def assign_groups(boxes: List[Box]) -> None:
    categories = sorted((b for b in boxes if b.kind == "category"), key=lambda b: b.ycenter)
    for b in boxes:
        if b.kind != "item":
            continue
        best = None
        for cat in categories:
            if cat.ycenter < b.ycenter:
                best = cat
            else:
                break
        b.group = best.idx if best is not None else None


# --------------------------------------------------------------------------
# Debug image
# --------------------------------------------------------------------------
def draw_debug(image: np.ndarray, boxes: List[Box]) -> np.ndarray:
    out = image.copy()
    by_id = {b.idx: b for b in boxes}

    # pairing lines first (background layer)
    for b in boxes:
        if b.role == "item_name" and b.paired_with is not None:
            other = by_id[b.paired_with]
            p1 = (int(b.xmax), int(b.ycenter))
            p2 = (int(other.xmin), int(other.ycenter))
            cv2.line(out, p1, p2, (255, 120, 0), 2)

    colors = {
        "noise": (150, 150, 150),
        "category": (0, 140, 255),
        "item_paired": (0, 200, 0),
        "unresolved": (0, 0, 255),
    }

    for b in boxes:
        pts = b.poly.astype(int)
        if b.kind == "noise":
            color = colors["noise"]
        elif b.kind == "category":
            color = colors["category"]
        elif b.kind == "item" and b.paired_with is not None:
            color = colors["item_paired"]
        else:
            color = colors["unresolved"]

        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
        label_pt = (int(b.xmin), max(int(b.ymin) - 6, 10))
        cv2.putText(out, str(b.idx), label_pt, cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, str(b.idx), label_pt, cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return out


# --------------------------------------------------------------------------
# Terminal report
# --------------------------------------------------------------------------
def print_report(boxes: List[Box]) -> None:
    by_id = {b.idx: b for b in boxes}
    categories = sorted((b for b in boxes if b.kind == "category"), key=lambda b: b.ycenter)

    print()
    if not categories:
        print("(no category headers detected)")

    for cat_num, cat in enumerate(categories, start=1):
        print(f"Category {cat_num}:")
        print(f"  box {cat.idx}")
        print("  items / prices")

        members = [b for b in boxes if b.kind == "item" and b.group == cat.idx]
        names = sorted(
            (b for b in members if b.role == "item_name"),
            key=lambda b: b.ycenter,
        )
        printed = set()
        for name in names:
            price = by_id[name.paired_with]
            print(f"  box {name.idx} ------- box {price.idx}")
            printed.add(name.idx)
            printed.add(price.idx)

        leftover = [b.idx for b in members if b.idx not in printed]
        if leftover:
            print(f"  (unresolved in this category: {leftover})")
        print()

    ungrouped_items = [b for b in boxes if b.kind == "item" and b.group is None]
    if ungrouped_items:
        print("No category / unassigned:")
        names = sorted(
            (b for b in ungrouped_items if b.role == "item_name"),
            key=lambda b: b.ycenter,
        )
        printed = set()
        for name in names:
            price = by_id[name.paired_with]
            print(f"  box {name.idx} ------- box {price.idx}")
            printed.add(name.idx)
            printed.add(price.idx)
        leftover = [b.idx for b in ungrouped_items if b.idx not in printed]
        if leftover:
            print(f"  (unresolved: {leftover})")
        print()

    noise = [b.idx for b in boxes if b.kind == "noise"]
    if noise:
        print(f"Noise / excluded boxes: {noise}")
    print()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run(image_path: str, out_path: str) -> List[Box]:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_area = float(image.shape[0] * image.shape[1])

    boxes = detect_boxes(image)
    median_h = classify_boxes(boxes, image_area)
    run_pairing_hungarian(boxes, median_h)
    assign_groups(boxes)

    debug_image = draw_debug(image, boxes)
    cv2.imwrite(out_path, debug_image)
    print(f"Debug image written to {out_path}")

    print_report(boxes)
    return boxes


def main():
    parser = argparse.ArgumentParser(description="Category/item/price detection experiment")
    parser.add_argument("image_path", help="Path to the menu image")
    parser.add_argument("--out", default="experiment2_debug.png", help="Path to write the debug image")
    args = parser.parse_args()

    run(args.image_path, args.out)


if __name__ == "__main__":
    main()