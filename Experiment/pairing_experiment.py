"""
pairing_experiment.py

STANDALONE test harness for the "tilted horizontal line" item<->price pairing
idea, kept completely separate from the real pipeline (detection.py /
recognition.py / config.py etc are NOT imported here on purpose, so you can
iterate on the algorithm without touching pipeline code).

What it does
------------
1. Gets text boxes for the menu image.
   - Tries PaddleOCR's TextDetection directly (same model the real pipeline
     uses) if paddleocr is importable and a network/model cache is available.
   - Otherwise falls back to a simple classical-CV box finder
     (threshold + morphological merge + contours) so the algorithm can still
     be exercised offline. This fallback is NOT meant to replace PaddleOCR in
     production -- it's here purely so this experiment script runs anywhere.

2. Classifies boxes as "category" (big font headers) vs "item/price"
   (everything else), using a height-based heuristic.

3. Runs the pairing algorithm exactly as you described:
   - draw a horizontal line through each candidate box's right-center point
   - find another box to the right that the line collides with -> paired
   - if nobody collides (orphan), tilt the line a bit (+1, -1, +2, -2, ...
     degrees) and retry, up to MAX_TILT_DEG
   - first match wins; box is marked used so it can't be paired twice

4. Groups every item/price pair (and unpaired boxes) under the nearest
   category header above it.

5. Prints a plain-text report:

    category 1: COFFEE (box 2)
        box 3 <-> box 4
        box 5 <-> box 6
        ...
    ORPHANS (no pair found within +/-MAX_TILT_DEG):
        box 9

6. Saves a color-coded debug image (pairing_debug_output.png):
    green      = item/price box
    orange     = category box
    gray       = classified as decorative/non-text noise, excluded
    red thick  = box that ended up an orphan (no pair found)
    blue->orange line = a successful pairing ray (blue = 0deg tilt,
                 shifting toward orange the more it had to tilt)
    dashed cyan = detected vertical divider lines (classical CV,
                 informational only for now -- see NOTES at bottom of file)

Usage
-----
    python pairing_experiment.py /path/to/menu.jpeg
    python pairing_experiment.py /path/to/menu.jpeg --force-classical
"""

import sys
import argparse
import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Tunables -- these are exactly the knobs you'll want to fiddle with while
# testing, so they're all up here instead of buried in the logic.
# --------------------------------------------------------------------------
MAX_TILT_DEG = 8          # how far we'll tilt the line before giving up
CATEGORY_HEIGHT_RATIO = 1.35   # a box taller than (median_height * this) is a
                                # category candidate
NOISE_HEIGHT_RATIO = 4.0       # a box taller than (median_height * this) is
                                # almost certainly a decorative/illustration
                                # blob, not a text line -- excluded entirely
MAX_VERTICAL_SHIFT_RATIO = 1.0  # the pairing line may drift at most this many
                                 # multiples of the median text height away
                                 # from its source row -- caps row-jumping
AMBIGUITY_MARGIN_DEG = 1.5      # if the best and second-best candidate need
                                 # tilt angles within this many degrees of
                                 # each other, treat it as a tie and defer to
                                 # the verification pass rather than guess
MIN_BOX_AREA = 40         # discard tiny speckle boxes (classical fallback only)


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass
class Box:
    idx: int
    points: np.ndarray  # 4x2
    kind: str = "unknown"      # "category" | "item" | "unknown"
    paired_with: Optional[int] = None
    pair_tilt_deg: Optional[float] = None
    verified_pass: bool = False  # True if resolved by the pass-2 re-verification step
    group: Optional[int] = None  # index into categories list

    @property
    def xmin(self):
        return float(self.points[:, 0].min())

    @property
    def xmax(self):
        return float(self.points[:, 0].max())

    @property
    def ymin(self):
        return float(self.points[:, 1].min())

    @property
    def ymax(self):
        return float(self.points[:, 1].max())

    @property
    def xcenter(self):
        return (self.xmin + self.xmax) / 2

    @property
    def ycenter(self):
        return (self.ymin + self.ymax) / 2

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def width(self):
        return self.xmax - self.xmin


# --------------------------------------------------------------------------
# Step 1a: real detector (PaddleOCR), used if available
# --------------------------------------------------------------------------
def detect_boxes_paddle(image: np.ndarray) -> list:
    from paddleocr import TextDetection  # noqa: local import, may not exist

    # enable_mkldnn=False works around a known bug in PaddlePaddle 3.3.x's
    # CPU inference backend (oneDNN/PIR executor) that throws
    # "NotImplementedError: ConvertPirAttribute2RuntimeAttribute not
    # support [...]" on CPU inference with MKL-DNN enabled (the default).
    # See: github.com/PaddlePaddle/Paddle/issues/77340
    detector = TextDetection(model_name="PP-OCRv5_mobile_det", enable_mkldnn=False)
    output = detector.predict(input=image, batch_size=1)
    result_item = next(iter(output))

    try:
        raw = list(result_item["dt_polys"])
    except (KeyError, TypeError):
        raw = list(result_item["res"]["dt_polys"])

    boxes = []
    for i, poly in enumerate(raw):
        pts = np.array(poly, dtype="float32")
        if cv2.contourArea(pts) < MIN_BOX_AREA:
            continue
        boxes.append(Box(idx=len(boxes), points=pts))
    return boxes


# --------------------------------------------------------------------------
# Step 1b: classical-CV fallback detector (offline-friendly, approximate)
#
# NOT a replacement for PaddleOCR -- just enough to exercise the pairing
# algorithm when paddleocr/model download isn't available in a given
# environment. Groups characters into word/line blobs via horizontal
# dilation, same rough idea as MSER-based text detectors.
# --------------------------------------------------------------------------
def _try_polarity(gray: np.ndarray, invert: bool, kernel, min_area: float):
    """Threshold with a given polarity, merge into words/lines, return the
    resulting boxes plus a 'plausibility' score used to pick the better
    polarity automatically."""
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, thresh = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    merged = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = gray.shape[0] * gray.shape[1]
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if h < 8 or w < 8:
            continue
        if area > img_area * 0.3:
            # a single blob covering a big chunk of the page means we almost
            # certainly picked the wrong polarity (background got selected
            # as "foreground") rather than found one giant text region
            continue
        pts = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype="float32"
        )
        boxes.append(Box(idx=len(boxes), points=pts))

    # Plausibility score: real text detection on a real menu produces many
    # small-ish, roughly word/line-shaped boxes. A wrong-polarity threshold
    # tends to produce either ~0 boxes (everything filtered as too big/small)
    # or a handful of huge ones. Score = count of boxes in a sane size band.
    score = sum(
        1 for b in boxes
        if 8 <= b.height <= gray.shape[0] * 0.08 and b.width <= gray.shape[1] * 0.6
    )
    return boxes, score


def detect_boxes_classical(image: np.ndarray) -> list:
    """Offline-friendly stand-in for a real text detector. NOT a replacement
    for PaddleOCR -- classical threshold+morphology cannot handle uneven
    lighting, shadows, paper texture, or real cursive handwriting the way a
    trained detector can. This exists purely so the pairing algorithm can be
    exercised without model downloads/network access.

    Two things that broke on other menu photos are handled here:
      1. Polarity (dark text on light bg vs light text on dark bg) is no
         longer assumed -- both are tried and the more plausible one wins.
      2. Kernel/area thresholds scale with image resolution instead of being
         fixed pixel constants tuned to one 736x1308 image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # scale morphology + min-area to this image's resolution instead of
    # hardcoding pixel counts that only made sense for the original menu
    kernel_w = max(9, int(w * 0.02))
    kernel_h = max(3, int(h * 0.004))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    min_area = max(20, (h * w) * 0.00002)

    boxes_inv, score_inv = _try_polarity(gray, invert=True, kernel=kernel, min_area=min_area)
    boxes_norm, score_norm = _try_polarity(gray, invert=False, kernel=kernel, min_area=min_area)

    if score_inv >= score_norm:
        chosen, polarity = boxes_inv, "dark text on light background"
    else:
        chosen, polarity = boxes_norm, "light text on dark background"
    print(f"[info] classical fallback: detected polarity = {polarity} "
          f"(inv candidates={len(boxes_inv)}, norm candidates={len(boxes_norm)})")

    for i, b in enumerate(chosen):
        b.idx = i
    return chosen


def get_boxes(image: np.ndarray, force_classical: bool) -> tuple[list, str]:
    if not force_classical:
        try:
            return detect_boxes_paddle(image), "paddleocr"
        except Exception as e:
            print(f"[info] PaddleOCR unavailable ({e}); falling back to classical CV detector.")
    return detect_boxes_classical(image), "classical"


# --------------------------------------------------------------------------
# Step 2: sort into reading order + classify category vs item
# --------------------------------------------------------------------------
def assign_reading_order(boxes: list, row_tolerance: float = 18.0) -> list:
    boxes = sorted(boxes, key=lambda b: (round(b.ycenter / row_tolerance), b.xcenter))
    for i, b in enumerate(boxes):
        b.idx = i
    return boxes


def classify_categories(boxes: list) -> float:
    if not boxes:
        return 0.0

    heights = sorted(b.height for b in boxes)
    n = len(heights)
    # trim the top ~20% before taking the median, so a handful of big
    # illustration blobs don't drag the "typical text height" estimate up
    trimmed = heights[: max(1, int(n * 0.8))]
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

    return median_h


# --------------------------------------------------------------------------
# Step 3: the tilt-search pairing algorithm
# --------------------------------------------------------------------------
def line_y_at_x(y0: float, x0: float, x: float, angle_deg: float) -> float:
    return y0 + (x - x0) * math.tan(math.radians(angle_deg))


def find_pair(source: Box, candidates: list, used: set, max_vertical_shift: float) -> list:
    """Rank every valid candidate to the right of `source`.

    IMPORTANT: this does NOT walk a fixed list of preset angles and stop at
    the first hit. An earlier version did that (0, +1, -1, +2, -2, ...) and
    it has an ordering bug: if the correct match needs a small *upward* tilt
    but a WRONG candidate (e.g. belonging to the row below) happens to be
    reachable with an even smaller *downward* tilt, the wrong one wins
    simply because "down" was checked first at that step. On a column of
    evenly-spaced rows this reliably cascades: one row grabs its neighbor's
    price, shifting everything below it down by one, leaving both ends of
    the column orphaned.

    Instead: for every candidate, solve directly for the angle that would
    make the ray pass exactly through its center, and rank ALL valid
    candidates by |angle| (smallest first). Returning the full ranked list
    (not just the winner) lets the caller detect genuine ties/ambiguity
    instead of silently committing to a coin flip.
    """
    x0, y0 = source.xmax, source.ycenter
    ranked = []

    for cand in candidates:
        if cand.idx == source.idx or cand.idx in used:
            continue
        if cand.kind != "item":
            continue
        if cand.xmin <= source.xmax:
            continue  # must be strictly to the right

        dx = cand.xcenter - x0
        dy = cand.ycenter - y0
        if abs(dy) > max_vertical_shift:
            continue  # candidate's own row is too far away, regardless of angle

        angle = math.degrees(math.atan2(dy, dx))
        if abs(angle) > MAX_TILT_DEG:
            continue

        ranked.append((abs(angle), cand.xmin - source.xmax, cand, round(angle, 1)))

    ranked.sort(key=lambda t: (t[0], t[1]))
    return ranked  # list of (abs_angle, x_dist, candidate, signed_angle), best first


def estimate_row_pitch(boxes: list) -> float:
    """Median gap between vertically-distinct item rows -- used to size the
    vertical-shift cap relative to this menu's actual line spacing instead
    of a fixed constant."""
    ys = sorted(b.ycenter for b in boxes if b.kind == "item")
    diffs = [b - a for a, b in zip(ys, ys[1:]) if b - a > 2]
    if not diffs:
        return 30.0
    diffs.sort()
    return diffs[len(diffs) // 2]


def _commit_pair(a: Box, b: Box, angle, used: set, verified: bool = False) -> None:
    a.paired_with = b.idx
    a.pair_tilt_deg = angle
    a.verified_pass = verified
    b.paired_with = a.idx
    b.pair_tilt_deg = angle
    b.verified_pass = verified
    used.add(a.idx)
    used.add(b.idx)


def run_pairing(boxes: list, median_item_height: float) -> None:
    """Two passes:

    Pass 1 commits only UNAMBIGUOUS matches -- if the best and second-best
    candidate require nearly the same tilt magnitude, that's a genuine tie
    (e.g. an item sitting almost exactly halfway between two price rows) and
    committing to either one risks being wrong, so it's deferred instead of
    guessed.

    Pass 2 (the "re-verification" pass): using the item<->price vertical
    offset learned from pass 1's confident matches, every remaining orphan
    is matched against the candidate whose position is closest to *where
    the price should be* given that learned offset -- instead of purely
    "closest by angle" -- which is exactly what breaks the ties pass 1
    correctly refused to guess at.
    """
    row_pitch = estimate_row_pitch(boxes)
    max_vertical_shift = max(median_item_height * MAX_VERTICAL_SHIFT_RATIO, row_pitch * 0.6)
    items = [b for b in boxes if b.kind == "item"]
    used = set()
    confident_dys = []

    # --- pass 1: commit only clear, unambiguous matches ---
    for src in items:
        if src.idx in used:
            continue
        ranked = find_pair(src, items, used, max_vertical_shift)
        if not ranked:
            continue
        best_abs_angle, _, best_cand, best_angle = ranked[0]
        if len(ranked) > 1:
            second_abs_angle = ranked[1][0]
            if (second_abs_angle - best_abs_angle) < AMBIGUITY_MARGIN_DEG:
                continue  # too close to call -- leave for pass 2
        _commit_pair(src, best_cand, best_angle, used)
        confident_dys.append(best_cand.ycenter - src.ycenter)

    # learn the typical item-center -> price-center vertical offset from the
    # matches we were actually confident about
    if confident_dys:
        confident_dys.sort()
        learned_dy = confident_dys[len(confident_dys) // 2]
    else:
        learned_dy = 0.0

    # --- pass 2: re-verify remaining items using the learned offset ---
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
            angle = round(math.degrees(math.atan2(best.ycenter - src.ycenter, best.xmax - src.xmax)), 1)
            _commit_pair(src, best, angle, used, verified=True)


# --------------------------------------------------------------------------
# Step 4: group under nearest category header above
# --------------------------------------------------------------------------
def assign_groups(boxes: list) -> list:
    categories = sorted([b for b in boxes if b.kind == "category"], key=lambda b: b.ycenter)
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
# Step 5: vertical divider detection (classical CV) -- informational only
# for now, drawn on the debug image. See NOTES at bottom of file.
# --------------------------------------------------------------------------
def detect_vertical_dividers(image: np.ndarray) -> list:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=60, minLineLength=image.shape[0] * 0.05,
        maxLineGap=8,
    )
    verticals = []
    if lines is None:
        return verticals
    for l in lines[:, 0, :]:
        x1, y1, x2, y2 = l
        if abs(x1 - x2) < 4 and abs(y2 - y1) > 20:
            verticals.append((x1, y1, x2, y2))
    return verticals


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def print_report(boxes: list, categories: list) -> None:
    by_idx = {b.idx: b for b in boxes}
    print("=" * 60)
    print("PAIRING / GROUPING REPORT")
    print("=" * 60)

    for i, cat in enumerate(categories, start=1):
        members = [b for b in boxes if b.group == cat.idx]
        members_sorted = sorted(members, key=lambda b: b.ycenter)
        print(f"\ncategory {i}: box {cat.idx}")
        seen_pairs = set()
        for m in members_sorted:
            if m.paired_with is not None:
                pair_key = tuple(sorted((m.idx, m.paired_with)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                other = by_idx[m.paired_with]
                left, right = (m, other) if m.xcenter < other.xcenter else (other, m)
                tilt_note = f" (tilt {m.pair_tilt_deg}deg)" if m.pair_tilt_deg else ""
                print(f"    box {left.idx} <-> box {right.idx}{tilt_note}")
            else:
                print(f"    box {m.idx}  -- ORPHAN (no pair found within +/-{MAX_TILT_DEG} deg)")

    unassigned = [b for b in boxes if b.kind == "item" and b.group is None]
    if unassigned:
        print("\nUnassigned boxes (no category above them):")
        for b in unassigned:
            print(f"    box {b.idx}")

    noise_boxes = [b for b in boxes if b.kind == "noise"]
    if noise_boxes:
        print("\nLikely decorative/non-text blobs (excluded from pairing, drawn gray):")
        for b in noise_boxes:
            print(f"    box {b.idx}  (h={b.height:.0f}px, w={b.width:.0f}px)")


# --------------------------------------------------------------------------
# Debug visualization
# --------------------------------------------------------------------------
def draw_debug(image: np.ndarray, boxes: list, verticals: list) -> np.ndarray:
    out = image.copy()
    by_idx = {b.idx: b for b in boxes}
    drawn_pairs = set()

    # vertical dividers (background layer) -- dashed cyan, deliberately a
    # different hue family from the pairing lines below so the two can't be
    # confused with each other
    for (x1, y1, x2, y2) in verticals:
        length = max(1, int(math.hypot(x2 - x1, y2 - y1)))
        steps = max(1, length // 10)
        for s in range(0, steps, 2):  # dashed: draw every other segment
            t0, t1 = s / steps, min(1.0, (s + 1) / steps)
            p0 = (int(x1 + (x2 - x1) * t0), int(y1 + (y2 - y1) * t0))
            p1 = (int(x1 + (x2 - x1) * t1), int(y1 + (y2 - y1) * t1))
            cv2.line(out, p0, p1, (255, 255, 0), 2)  # cyan

    # pairing lines -- blue at 0deg tilt, shifting to orange as tilt increases
    for b in boxes:
        if b.paired_with is None or b.kind == "category":
            continue
        pair_key = tuple(sorted((b.idx, b.paired_with)))
        if pair_key in drawn_pairs:
            continue
        drawn_pairs.add(pair_key)
        other = by_idx[b.paired_with]
        left, right = (b, other) if b.xcenter < other.xcenter else (other, b)
        angle = b.pair_tilt_deg or 0
        tilt_fraction = min(abs(angle) / MAX_TILT_DEG, 1.0)
        # BGR: blue (255,0,0) -> orange (0,140,255)
        color = (
            int(255 * (1 - tilt_fraction)),
            int(140 * tilt_fraction),
            int(255 * tilt_fraction),
        )
        p1 = (int(left.xmax), int(left.ycenter))
        p2 = (int(right.xmin), int(line_y_at_x(left.ycenter, left.xmax, right.xmin, angle)))
        cv2.line(out, p1, p2, color, 2)

    # boxes -- green normally, orange for category, thick red if orphan
    for b in boxes:
        pts = b.points.astype(int)
        if b.kind == "noise":
            color, thickness = (150, 150, 150), 2  # gray = likely decorative
        elif b.kind == "category":
            color, thickness = (0, 140, 255), 3   # orange
        elif b.paired_with is None:
            color, thickness = (0, 0, 255), 3      # red = orphan
        else:
            color, thickness = (0, 255, 0), 2       # green
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
        label_pt = tuple(pts[0])
        cv2.putText(out, str(b.idx), label_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--force-classical", action="store_true",
                         help="skip PaddleOCR even if importable")
    parser.add_argument("--out", default="pairing_debug_output.png")
    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Could not read image: {args.image_path}")
        sys.exit(1)

    boxes, source = get_boxes(image, args.force_classical)
    print(f"[info] detector used: {source}, boxes found: {len(boxes)}")

    if not boxes:
        print("[error] 0 boxes detected -- nothing to pair. If this was the "
              "classical fallback, the image likely isn't a good fit for "
              "threshold+morphology (low contrast, heavy texture/shadow, or "
              "a resolution/kernel mismatch). Try --force-classical off "
              "(use real PaddleOCR) or inspect the thresholded image directly.")
        sys.exit(1)

    boxes = assign_reading_order(boxes)
    median_item_height = classify_categories(boxes)
    run_pairing(boxes, median_item_height)
    categories = assign_groups(boxes)
    verticals = detect_vertical_dividers(image)

    print_report(boxes, categories)

    debug_img = draw_debug(image, boxes, verticals)
    cv2.imwrite(args.out, debug_img)
    print(f"\n[info] debug image saved to {args.out}")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# NOTES on false-positive / "garbage" boxes (e.g. the "Coffee" text baked
# into the cup illustration, or the little squiggle read as a box) --
# see the reply this script was delivered with for the full writeup. The
# short version: those detections are geometrically real (there IS ink
# there) but they don't belong to the item/price grid, so the cheapest
# fix is a post-detection *layout* filter rather than a detection-time
# fix: cluster item boxes by x-position into "column bands" (name column,
# price column) and drop anything that doesn't fall into an established
# band and also doesn't align on a row with two+ other boxes.
# --------------------------------------------------------------------------