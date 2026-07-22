# Pairing Experiment: Tilted Horizontal Line Algorithm

Standalone test harness for the "tilted horizontal line" item-to-price pairing algorithm. This experiment is kept completely separate from the main pipeline to allow rapid iteration on the pairing logic.

## What It Does

The algorithm pairs text regions (item names with prices) using a geometric ray-casting approach:

### Step-by-Step Algorithm

1. **Text Detection**
   - Uses PaddleOCR's TextDetection model (same as production pipeline)
   - Falls back to classical CV (threshold + morphology) if PaddleOCR unavailable
   - Returns bounding boxes for all text regions

2. **Reading Order Assignment**
   - Sorts boxes top-to-bottom, left-to-right
   - Groups into horizontal "rows" using Y-coordinate tolerance

3. **Box Classification**
   - Calculates median text height across all boxes
   - **Category headers**: boxes taller than `median × 1.35`
   - **Items/Prices**: normal-height boxes
   - **Noise**: boxes taller than `median × 4.0` (decorative elements, illustrations)

4. **Pairing Algorithm** (the core innovation)
   
   For each unpaired item box:
   - Draw a **horizontal line** through the box's right-center point
   - Look for another box to the right where the line collides
   - **If no collision**: tilt the line in small increments (±1°, ±2°, ..., up to ±8°)
   - **First match wins**, both boxes marked as paired
   - Prevents double-pairing (each box can only pair once)
   - **Vertical drift limit**: line can't shift more than `median_height × 1.0` away from source row
     - Prevents long gaps + small angles from grabbing boxes in different rows

5. **Grouping**
   - Assigns each item/price pair to the nearest category header above it
   - Creates hierarchical menu structure

6. **Visualization**
   - **Green boxes**: successfully paired items/prices
   - **Orange boxes**: category headers
   - **Red boxes**: orphans (no pair found within ±8°)
   - **Gray boxes**: noise/decorative elements (excluded from pairing)
   - **Blue → Orange lines**: pairing rays (blue = 0° tilt, orange = max tilt)
   - **Dashed cyan lines**: vertical dividers detected (informational only)

## Tunable Parameters

All tuning knobs are at the top of `pairing_experiment.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_TILT_DEG` | 8 | Maximum angle to tilt the pairing line before giving up |
| `TILT_STEP_DEG` | 1 | *(Deprecated)* No longer used - algorithm solves for exact angles |
| `CATEGORY_HEIGHT_RATIO` | 1.35 | Box taller than median × this is a category header |
| `NOISE_HEIGHT_RATIO` | 4.0 | Box taller than median × this is decorative noise |
| `MAX_VERTICAL_SHIFT_RATIO` | 1.0 | Max vertical drift as multiple of median height |
| `AMBIGUITY_MARGIN_DEG` | 1.5 | Angle difference threshold for detecting ties (Pass 1 defers, Pass 2 resolves) |
| `MIN_BOX_AREA` | 40 | Discard tiny speckle boxes (classical fallback only) |

## Usage

### Single Image
```bash
python pairing_experiment.py path/to/menu.jpg
python pairing_experiment.py path/to/menu.jpg --force-classical
python pairing_experiment.py path/to/menu.jpg --out my_output.png
```

### Batch Processing
```bash
python batch_experiment.py
python batch_experiment.py --force-classical
python batch_experiment.py --input-dir ../data/samples --output-dir results
```

Batch mode:
- Processes all images in `input/` folder
- Saves debug visualizations to `debug_output/`
- Prints per-image reports + aggregate statistics
- Shows pairing success rate and orphan rate

## How It Works (Technical Details)

### Algorithm Architecture: Two-Pass Approach

The pairing algorithm uses **two passes** to avoid a critical ordering bug found in simpler implementations:

**Pass 1: Commit only unambiguous matches**
- For each source box, compute the exact angle needed to reach every candidate
- Rank all candidates by absolute angle (smallest first)
- **Critical**: If the best and second-best candidates require angles within `AMBIGUITY_MARGIN_DEG` (1.5°), it's a genuine tie → defer to Pass 2
- Only commit clear winners (significant angle difference between best and second-best)
- Track the vertical offset (dy) between paired items and prices

**Pass 2: Re-verification using learned offset**
- Calculate median dy from Pass 1's confident matches
- For remaining orphans, predict where the price **should be** based on learned offset
- Match to candidate closest to predicted position (not closest by angle)
- Breaks ties that Pass 1 correctly refused to guess

### Why Two Passes?

**The Ordering Bug (fixed by this approach)**:

Early versions tried angles in sequence `[0, +1, -1, +2, -2, ...]` and stopped at first hit. This has a fatal flaw:

```
Row 1: [Item A ______]            (correct price: 3.50, needs +1° tilt)
Row 2: [Item B ___] [3.50]        (B's correct price)
Row 3: [Item C ___] [4.00]        (C's correct price, reachable at -2° from A)
```

If Item A's correct match (3.50 at +1°) is checked AFTER a wrong match (4.00 at -2°), the wrong one wins simply due to search order. This cascades down the column: one row "steals" the next row's price, everything shifts down, both ends become orphans.

**Solution**: Don't iterate through preset angles. Instead, solve directly for the angle to each candidate, rank ALL candidates by |angle|, and detect genuine ambiguity (ties) rather than making coin flips based on search order.

### Ray Casting Logic (Pass 1)
```python
# For each candidate, solve for the exact angle needed
dx = candidate.xcenter - source.xmax
dy = candidate.ycenter - source.ycenter
angle = atan2(dy, dx)  # Direct solve, not sequential search

# Rank ALL valid candidates by absolute angle
ranked.sort(key=lambda: abs(angle))

# Detect ties: if best and second-best are within 1.5°, defer
if len(ranked) > 1:
    if abs(ranked[1].angle - ranked[0].angle) < AMBIGUITY_MARGIN_DEG:
        # Too close to call → leave for Pass 2
        continue
```

### Vertical Drift Prevention

Two mechanisms prevent cross-row pairing:

**1. Candidate Pre-Filter**: 
```python
if abs(candidate.ycenter - source.ycenter) > max_vertical_shift:
    skip  # Candidate's own row is too far, regardless of angle
```

**2. Adaptive Shift Limit**:
```python
row_pitch = estimate_row_pitch(boxes)  # Median gap between rows
max_vertical_shift = max(
    median_item_height * 1.0,  # At least 1× text height
    row_pitch * 0.6             # Or 60% of typical row spacing
)
```

This adapts to each menu's actual line spacing instead of using a fixed constant.

**Example of why this matters**:
```
Row 1:  [Coffee ________________]   ← Long gap
Row 2:  [Tea _____] [3.50]
Row 3:  [Juice _______________] [4.00]
```

Without vertical shift limit: Coffee + 2° tilt over long distance → reaches 4.00 (wrong row)  
With limit: Coffee can't drift more than `row_pitch × 0.6` → rejects 4.00, finds correct match or stays orphan

### Why Tilted Lines?

Real-world menu photos have imperfections:
- Camera held at slight angle
- Paper not perfectly flat  
- Handwriting naturally slopes
- Table surface not level

A strict horizontal line (0°) would miss many valid pairs. The tilt-search allows small deviations while preventing wild mismatches.

## Blind Spots & Limitations

### 1. **Vertical/Multi-Column Layouts**
**Problem**: Algorithm assumes horizontal left-to-right pairing (item on left, price on right).

**Breaks on**:
- Vertical price lists (price above/below item)
- Three-column layouts (item | size | price)
- Mixed layouts (some sections horizontal, others vertical)

**Example failure**:
```
┌─────────────┐
│ Coffee      │
│ 3.50        │  ← Price directly below item (vertical)
│             │
│ Tea    4.00 │  ← Horizontal pairing works here
└─────────────┘
```

### 2. **False-Positive Text Regions**
**Problem**: Detector finds text-like shapes that aren't part of the menu grid.

**Common false positives**:
- Text baked into illustrations (e.g., "Coffee" printed on a cup graphic)
- Decorative squiggles misread as text
- Watermarks, stamps, page numbers
- Border patterns with text-like texture

**Current mitigation**: `NOISE_HEIGHT_RATIO` filters very tall boxes, but doesn't catch everything.

**Better solution** (not implemented): Cluster boxes by X-position into "column bands" (name column, price column). Drop anything outside established bands that doesn't align with 2+ other boxes on the same row.

### 3. **Ambiguous Layouts**
**Problem**: When multiple boxes are valid collision candidates, algorithm picks the nearest one (shortest horizontal distance). This fails if layout is ambiguous.

**Example failure**:
```
[Espresso] [Single 2.50] [Double 3.50]
```
"Espresso" will pair with "Single 2.50" (nearest), but user might mean "Double 3.50" or expect two separate items.

### 4. **Dense Packed Text**
**Problem**: When text regions are tightly packed, the detector may merge them into a single box.

**Example failure**:
```
[Coffee Tea Juice 3.50 4.00 5.00]  ← Entire row merged into one box
```
Can't pair items if detector doesn't separate them. Requires better text detection preprocessing.

### 5. **Missing Prices / Orphan Items**
**Problem**: Algorithm gives up after `MAX_TILT_DEG`. If the actual angle exceeds this, or if there genuinely is no price, the item becomes an orphan.

**Mitigation**: Tune `MAX_TILT_DEG` higher, but this increases false-match risk (pairing across different rows).

### 6. **Double-Pairing Prevention Side Effects**
**Problem**: In simpler first-match-wins implementations, a greedy item can "steal" a price that belongs to a later item.

**Status**: **MITIGATED** by two-pass algorithm. The current implementation:
- Pass 1 only commits unambiguous matches (clear angle separation)
- Pass 2 uses learned vertical offset to match remaining items to predicted price positions
- Reduces greedy-stealing dramatically, though edge cases may still exist

**Remaining edge case**:
```
Row 1: [Coffee _____________________]
Row 2: [Tea ___] [3.50]
```
If Coffee → 3.50 has clear angle advantage in Pass 1 despite wrong semantics, it still wins.

**Ultimate solution**: Global optimization (Hungarian algorithm) over all possible pairings, scored by distance + angle + learned offset. Would eliminate all greedy-matching artifacts.

### 7. **Rotated/Skewed Images**
**Problem**: If the entire menu is rotated >10°, even `MAX_TILT_DEG=8` won't compensate.

**Solution**: The main pipeline has a deskew step (preprocessing) that aligns the image before text detection. This experiment doesn't include that step.

### 8. **Classical Fallback Limitations**
**Problem**: The classical CV detector (threshold + morphology) cannot handle:
- Uneven lighting / shadows
- Low contrast text
- Colored backgrounds
- Cursive handwriting
- Paper texture noise

**Mitigation**: The fallback now auto-detects polarity (dark-on-light vs light-on-dark) and scales morphology kernels to image resolution instead of using fixed pixel constants. This makes it work across a wider range of menus than early fixed-threshold versions.

**Polarity Auto-Detection**:
```python
# Try both polarities, score by number of text-shaped boxes found
boxes_inverted, score_inv = threshold_with_polarity(invert=True)
boxes_normal, score_norm = threshold_with_polarity(invert=False)

# Pick the polarity that produces more plausible text regions
chosen = boxes_inverted if score_inv >= score_norm else boxes_normal
```

**Plausibility scoring**: Real text detection produces many small-ish, word/line-shaped boxes. Wrong polarity produces either ~0 boxes (everything filtered) or a few huge blobs (background selected as foreground).

**Purpose**: The classical fallback exists purely for offline testing when PaddleOCR isn't available. It's **not a replacement** for a trained detector in production.

## When Code Will Break

### Hard Failures (Crash/Exception)
- Image file doesn't exist or is corrupted
- Image has 0 dimensions or wrong format
- PaddleOCR model files missing (will fallback to classical)
- Out of memory (very large images)

### Soft Failures (Poor Results)
- Menus with vertical item-price layout
- Three+ column layouts
- Heavily rotated images (>10° without deskew)
- Very low resolution (<500px on shortest side)
- Text and background have similar brightness (low contrast)
- Many decorative elements misdetected as text

### Edge Cases
- Single item with no price → orphan (expected)
- All items in a row → will try to pair them sequentially (first-to-second, third-to-fourth)
- Price on left, item on right → will pair backward (algorithm doesn't validate semantic meaning)
- Prices without currency symbols → works (just pairs geometric boxes)

## Comparison with Production Pipeline

| Feature | Experiment | Production Pipeline |
|---------|------------|---------------------|
| **Detection** | PaddleOCR + classical fallback | PaddleOCR only |
| **Preprocessing** | None | Resize, deskew, denoise, CLAHE |
| **Pairing** | Two-pass tilted line (geometric) | Y-distance nearest neighbor |
| **Column Detection** | None | Automatic left/right split |
| **Price Extraction** | None | Regex + OCR error correction |
| **Currency Detection** | None | TND/EUR/€ with fallback logic |
| **Validation** | None | File size, format, dimensions |
| **Error Handling** | None | Custom exception hierarchy |
| **Logging** | Print statements | Structured logging |
| **Thread Safety** | Not applicable | Lock-protected singletons |

**Key difference**: The experiment focuses purely on the geometric pairing algorithm, while the production pipeline is a complete end-to-end system with validation, logging, error handling, and production-ready features.

## Implementation Insights

### Data Structure Choice
Uses `dataclass Box` with computed properties (`@property`) for geometric queries:
- Clean separation: raw points stored, derived values computed on-demand
- No cache invalidation bugs (points never change after creation)
- Readable: `box.xcenter` vs `(box.xmin + box.xmax) / 2` everywhere

### Reading Order Assignment
```python
boxes = sorted(boxes, key=lambda b: (round(b.ycenter / row_tolerance), b.xcenter))
```
- **Two-level sort**: Primary by row (Y), secondary by column (X)
- **row_tolerance=18px**: Boxes within 18px vertically are considered "same row"
- **round() trick**: Discretizes Y-coordinates into row "bins" without explicit clustering
- Simple but effective for well-formatted menus

### Median Calculation with Trimming
```python
heights = sorted(b.height for b in boxes)
trimmed = heights[: int(len(heights) * 0.8)]  # Drop top 20%
median_h = trimmed[len(trimmed) // 2]
```
**Why trim?** A few large decorative elements (illustrations, logos) would drag the median up, making normal text look "small" by comparison. Trimming the top 20% gives a robust estimate of typical text height.

### Adaptive Row Pitch Estimation
```python
def estimate_row_pitch(boxes):
    ys = sorted(b.ycenter for b in boxes if b.kind == "item")
    diffs = [b - a for a, b in zip(ys, ys[1:]) if b - a > 2]
    return median(diffs)
```
Learns the menu's actual line spacing instead of assuming a fixed constant. Used to set `max_vertical_shift = max(text_height, row_pitch × 0.6)`.

### Two-Pass Commit Function
```python
def _commit_pair(a, b, angle, used, verified=False):
    a.paired_with = b.idx
    b.paired_with = a.idx
    a.verified_pass = verified  # Tracks which pass resolved this pair
    used.add(a.idx)
    used.add(b.idx)
```
The `verified_pass` flag enables future analysis: which pairs were obvious (Pass 1) vs ambiguous and required learned offset (Pass 2)?

### Performance Characteristics
- **Time Complexity**: O(n²) per pass (each of n items checks n candidates)
- **Space Complexity**: O(n) for boxes + ranked lists
- **Typical runtime**: <100ms for 40-50 text boxes on modern CPU
- **Bottleneck**: Not the pairing algorithm itself, but text detection (PaddleOCR: 5-15 seconds)

## Future Improvements

### High Impact
1. **Column clustering filter**: Reject false-positive boxes outside established column bands
2. **Global optimization (Hungarian algorithm)**: Replace greedy two-pass with optimal bipartite matching across ALL item-price pairs simultaneously
3. **Multi-layout detection**: Auto-detect vertical vs horizontal layout, switch algorithms

### Medium Impact
4. **Confidence scoring**: Weight pairings by distance + angle + offset deviation, flag low-confidence pairs
5. **Semantic validation**: Use OCR text to confirm item-price pairs make sense (text on left, number on right)
6. **Adaptive parameter tuning**: Learn optimal `MAX_TILT_DEG` / `CATEGORY_HEIGHT_RATIO` per image based on detected layout

### Low Impact
7. **Better visualization**: Color-code by confidence, show rejected candidates, highlight Pass 2 re-verified pairs
8. **Export structured results**: Save JSON output (not just visualization) with pair metadata
9. **Row clustering**: Group boxes into explicit row objects before pairing (currently implicit via Y-coordinate sorting)

### Algorithm Evolution Notes

**Current State (v2 - Two-Pass)**:
- ✅ Fixed sequential angle search ordering bug
- ✅ Detects and defers genuine ambiguity (ties)
- ✅ Learns vertical offset from confident matches
- ✅ Adaptive vertical shift limit based on row pitch
- ⚠️ Still greedy within each pass (local optimization)

**Next Evolution (v3 - Global Optimization)**:
- Formulate as bipartite matching problem
- Score all possible item-price pairings: `score = α×angle + β×distance + γ×offset_deviation`
- Solve with Hungarian algorithm → globally optimal assignment
- Eliminates all greedy-matching artifacts
- Produces explicit confidence scores per pair

---

# Hungarian Algorithm Experiment: Global Optimization Approach

A more advanced pairing algorithm that uses **global optimization** instead of greedy local decisions. This represents the next evolution of the pairing logic - from sequential angle search (v1) → two-pass with ties detection (v2 - current production) → **Hungarian bipartite matching (v3 - this experiment)**.

## What experiment2.py Does

`experiment2.py` implements a **globally optimal** item-price pairing using the **Hungarian Algorithm** (scipy.optimize.linear_sum_assignment). Unlike the production pipeline's two-pass greedy approach, this finds the single best set of pairings across the entire menu simultaneously.

### Key Algorithmic Differences

| Approach | Production (`pairing.py`) | Experiment2 (`experiment2.py`) |
|----------|---------------------------|-------------------------------|
| **Pairing Method** | Two-pass greedy (local decisions) | Hungarian Algorithm (global optimization) |
| **Matching Strategy** | For each item, find its best price | Find best assignment across ALL items/prices at once |
| **Category Detection** | Single signal: height > median × 1.8 | **3-signal voting system** (needs 2/3 votes) |
| **Noise Detection** | Height only (> median × 4.0) | **Shape-based** (area + aspect ratio) |
| **Rejection Handling** | Leave unmatched after 2 passes | Cost-matrix padding with rejection threshold |
| **Optimization Scope** | Greedy within each pass | Globally optimal across entire menu |

### Algorithm Pipeline

**1. Text Detection**
- Uses PaddleOCR TextDetection (PP-OCRv5_mobile_det)
- No classical fallback (relies on trained detector)

**2. Noise Classification (Shape-Based)**

Unlike production's height-only filter, experiment2 uses **both size AND shape**:

```python
is_noise = (area_ratio >= 0.05) AND (aspect_ratio < 2.0)
```

**Why both conditions?**
- Illustrations: Large AND squarish/tall (both conditions true)
- Banner headlines: Large but wide (aspect > 2.0) → NOT noise
- Small logos: Squarish but tiny (area < 5%) → NOT noise

**Tunable:** `NOISE_AREA_RATIO`, `NOISE_ASPECT_MIN`

**3. Category Detection (3-Signal Voting)**

**Core Innovation:** Instead of a fragile height threshold that breaks when handwriting scale drifts, use **three independent geometric signals**:

**Vote (a): No companion in established price column**
- Detect repeated vertical alignments (columns) by binning box x-positions
- Check if this box has a row-mate at the price column's x-position
- If alone → +1 vote for category

**Vote (b): Row isolation**
- Does ANY other box overlap this row vertically?
- If nothing shares this horizontal band → +1 vote for category
- Works even without distinct columns (single-column layouts)

**Vote (c): Whitespace gap above**
- Measure gap to nearest box fully above
- Compare to page's median row-to-row spacing
- If gap > median_spacing × 1.6 → +1 vote for category

**Decision Rule:** Need **≥2 votes** (out of 3) to be classified as category

**Why this works:**
- No single fragile threshold decides anything
- Adapts to each menu's actual layout (learned row spacing, detected columns)
- Height-independent (works when categories aren't drawn bigger)
- Robust to handwriting scale drift across the page

**Tunables:**
- `COLUMN_X_TOLERANCE = 15` (px - how close = "same column")
- `MIN_COLUMN_COLLISIONS_FOR_ITEM = 2` (min boxes to count as "established column")
- `GAP_RATIO = 1.6` (gap > spacing × this → vote yes)
- `CATEGORY_VOTE_THRESHOLD = 2` (votes needed out of 3)

**4. Hungarian Algorithm Pairing**

**Cost Matrix Construction:**

For every pair (item_i, candidate_j):

```python
if structurally_invalid(i, j):  # wrong direction, too far, too tilted
    cost[i][j] = BIG  # 1e6 - effectively infinite
else:
    angle = atan2(dy, dx)
    cost[i][j] = W_ANGLE × |angle| + W_VDIST × |dy|
```

**Structural Validity Checks:**
- Must be strictly to the right (candidate.xmin > source.xmax)
- Vertical offset within limit: |dy| ≤ max(median_height × 1.2, row_pitch × 0.6)
- Tilt angle within bounds: |angle| ≤ 8°

**Rejection Threshold Padding:**

Standard assignment-with-rejection technique - pad NxN cost matrix to 2Nx2N:

```
[ real_costs     | REJECT_COST × ones ]
[ REJECT_COST    | zeros              ]
```

This gives Hungarian a "no match" option instead of forcing every item into a pair. Any real pairing with cost ≥ REJECT_COST is treated as "leave unresolved".

**Tunables:**
- `MAX_TILT_DEG = 8.0` (max angle between name/price)
- `MAX_VERTICAL_SHIFT_RATIO = 1.2` (max row-jumping, × median height)
- `REJECT_COST = 999.0` (threshold for "no match better than this")
- `W_ANGLE = 1.0` (cost weight for angle deviation)
- `W_VDIST = 0.08` (cost weight for vertical distance)
- `BIG = 1e6` (cost for structurally invalid pairs)

**Double-Use Resolution:**

The bipartite relaxation can assign the same box as both a "name" (row) and "price" (column). Post-process resolves this:

```python
accepted_pairs.sort(key=lambda: cost)  # Hungarian's globally-optimal order
used = set()
for cost, name_idx, price_idx in accepted_pairs:
    if name_idx not in used and price_idx not in used:
        commit_pair(name_idx, price_idx)
        used.add(name_idx)
        used.add(price_idx)
```

Greedy pass through Hungarian's cost-sorted results keeps best uses, drops conflicts.

**5. Grouping**
- Same as production: assign items to nearest category header above

**6. Debug Visualization**
- **Gray boxes**: Noise (excluded)
- **Orange boxes**: Category headers  
- **Green boxes**: Successfully paired items/prices
- **Red boxes**: Unresolved (no valid match found)
- **Orange lines**: Pairing connections

## Usage

```bash
python experiment2.py path/to/menu.jpg
python experiment2.py path/to/menu.jpg --out my_debug.png
```

Output:
- Debug image saved to `--out` path (default: `experiment2_debug.png`)
- Terminal report showing category groups and paired items

## Why Global Optimization Matters

### The Cascade Problem (Greedy Matching)

Production's two-pass algorithm makes **local** decisions:

```
Row 1: [Coffee _______________]     (needs 3.50 at +2° tilt)
Row 2: [Tea _____] [3.50]           (Tea's correct price)
Row 3: [Juice ___] [4.00]           (Juice's correct price)
```

**Greedy behavior (even with two passes):**
1. If Coffee's ambiguity detection fails, commits Coffee ↔ 3.50
2. Tea now has no valid candidate → orphan
3. Juice ↔ 4.00 pairs correctly
4. Result: 1 correct, 1 wrong, 1 orphan

**Hungarian behavior:**
1. Computes cost for ALL possible pairings:
   - Coffee ↔ 3.50: angle=2°, cost=2.0
   - Coffee ↔ 4.00: angle=5°, cost=5.0
   - Tea ↔ 3.50: angle=0°, cost=0.0
   - Juice ↔ 4.00: angle=1°, cost=1.0
2. Finds globally minimal total cost assignment
3. Result: Tea ↔ 3.50 (0.0), Juice ↔ 4.00 (1.0), Coffee orphan (correct - no valid match)

**Key advantage:** Doesn't commit to locally-reasonable but globally-suboptimal matches.

### When Hungarian Helps Most

- **Dense/crowded rows** where multiple items could plausibly pair with same price
- **Ambiguous geometries** where greedy first-match causes cascades
- **Missing prices** - leaves items unmatched instead of forcing wrong pairs
- **Tilted/skewed menus** where angle-sorting alone isn't enough

### When It Doesn't Matter

- Clean, well-spaced menus where greedy already works
- Single-column layouts with inline prices (no ambiguity)
- Menus where every item has exactly one nearby price candidate

## Processing Logic for Reliable Results

### Robustness Features

**1. Adaptive Thresholds**
- Row pitch learned from actual box spacing (not hardcoded)
- Column positions detected from alignment clustering
- Vertical shift limit scales with row pitch

**2. Multiple Independent Signals**
- Category detection uses 3 votes (not 1 threshold)
- Noise detection uses shape + size (not just size)
- Pairing uses angle + distance (not just angle)

**3. Rejection Over Forcing**
- Cost matrix padding allows "no match"
- Structurally invalid pairs get cost=BIG (never selected)
- Any real cost ≥ REJECT_COST treated as rejection

**4. Known Limitations**
- **Honest limitation:** An item with genuinely missing price and a category header look identical from geometry alone (both "alone on row"). Resolving needs recognized text (price pattern detection) - out of scope for geometry-only pass.
- **Vertical layouts:** Algorithm assumes horizontal left-to-right pairing (item → price)
- **Three+ columns:** Only detects leftmost/rightmost columns

## Performance Tuning Settings

### For Cleaner Results

**If too many false categories:**
```python
CATEGORY_VOTE_THRESHOLD = 3  # All 3 votes required (stricter)
GAP_RATIO = 2.0              # Bigger gap needed for whitespace vote
```

**If missing real categories:**
```python
CATEGORY_VOTE_THRESHOLD = 1  # Any single vote sufficient (looser)
COLUMN_X_TOLERANCE = 25      # Wider binning for price column detection
```

**If noise blobs getting through:**
```python
NOISE_AREA_RATIO = 0.03      # Lower threshold (catch smaller blobs)
NOISE_ASPECT_MIN = 2.5       # Higher aspect ratio (stricter shape requirement)
```

**If too many orphans (valid pairs rejected):**
```python
MAX_TILT_DEG = 12.0           # Allow more angle tolerance
MAX_VERTICAL_SHIFT_RATIO = 1.5  # Allow more row-jumping
REJECT_COST = 1500.0          # Higher threshold (accept worse matches)
```

**If wrong pairings (too permissive):**
```python
MAX_TILT_DEG = 6.0            # Stricter angle limit
REJECT_COST = 500.0           # Lower threshold (reject marginal matches)
W_ANGLE = 2.0                 # Penalize angle deviation more heavily
```

**If pairing across wrong rows:**
```python
MAX_VERTICAL_SHIFT_RATIO = 0.8  # Stricter vertical limit
W_VDIST = 0.15                  # Penalize vertical distance more
```

### Cost Weight Tuning

The cost function balances two factors:

```python
cost = W_ANGLE × |angle| + W_VDIST × |dy|
```

**Default:** `W_ANGLE=1.0, W_VDIST=0.08`

**Interpretation:**
- 1° of tilt ≈ 12.5px vertical offset (1.0 / 0.08)
- Algorithm prefers straighter lines over closer vertical alignment

**To prioritize vertical alignment:**
```python
W_ANGLE = 0.5   # Care less about angle
W_VDIST = 0.2   # Care more about vertical distance
```

**To prioritize straight horizontal lines:**
```python
W_ANGLE = 2.0   # Strongly penalize tilt
W_VDIST = 0.05  # Care less about vertical offset
```

**Tuning Strategy:**
1. Start with defaults
2. If getting cross-row pairings → increase `W_VDIST`
3. If getting far-but-aligned matches → increase `W_ANGLE`
4. If too many orphans → lower both weights slightly

## Comparison: Tilted Line vs Hungarian

| Aspect | Tilted Line (pairing_experiment.py) | Hungarian (experiment2.py) |
|--------|-------------------------------------|----------------------------|
| **Category Detection** | Height threshold only | 3-signal voting |
| **Noise Detection** | Height only | Shape-based (area + aspect) |
| **Pairing Scope** | Local (per-item greedy) | Global (all pairs optimized) |
| **Ambiguity Handling** | Two-pass with learned offset | Cost matrix with rejection |
| **Cascade Errors** | Possible (early bad match affects later) | Eliminated (global optimization) |
| **Performance** | O(n²) per pass | O(n³) Hungarian + O(n²) build |
| **Complexity** | ~200 lines | ~400 lines |
| **Best For** | Clean menus, rapid prototyping | Dense/ambiguous layouts, production quality |

**When to use each:**
- **Tilted Line:** Fast iteration, well-spaced menus, educational purposes
- **Hungarian:** Maximum accuracy, crowded layouts, production deployment

## Integration Path

To integrate experiment2 logic into production pipeline:

1. **Replace** `src/pairing.py::classify_boxes()` with experiment2's 3-signal voting
2. **Replace** `src/pairing.py::run_pairing()` with Hungarian cost-matrix approach
3. **Add** scipy as dependency in `requirements.txt`
4. **Tune** cost weights and thresholds on representative test set
5. **Validate** on batch_experiment test suite

**Migration risk:** Low - experiment2's `build_skeleton()` has same signature as production's, just different internal logic.

---

**Made with dedication by Hamza Z** ✨
