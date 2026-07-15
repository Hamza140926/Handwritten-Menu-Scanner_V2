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

**Made with dedication by Hamza Z** ✨
