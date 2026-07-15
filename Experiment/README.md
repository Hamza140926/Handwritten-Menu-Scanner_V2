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
| `TILT_STEP_DEG` | 1 | Degree increment per retry (±1°, ±2°, ...) |
| `CATEGORY_HEIGHT_RATIO` | 1.35 | Box taller than median × this is a category header |
| `NOISE_HEIGHT_RATIO` | 4.0 | Box taller than median × this is decorative noise |
| `MAX_VERTICAL_SHIFT_RATIO` | 1.0 | Max vertical drift as multiple of median height |
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

### Ray Casting Logic
```python
# 1. Start with horizontal line through box's right edge
y0 = source_box.ycenter
x0 = source_box.xmax

# 2. Try angle 0°, then ±1°, ±2°, ..., up to ±MAX_TILT_DEG
for angle in [0, +1, -1, +2, -2, ..., +8, -8]:
    # 3. Calculate line's Y coordinate at candidate's X center
    y_line = y0 + (candidate.xcenter - x0) * tan(angle)
    
    # 4. Check collision: does line pass through candidate box?
    if candidate.ymin <= y_line <= candidate.ymax:
        # 5. Check vertical drift limit
        if abs(y_line - y0) <= max_vertical_shift:
            # MATCH FOUND
            return candidate, angle
```

### Why Tilted Lines?

Real-world menu photos have imperfections:
- Camera held at slight angle
- Paper not perfectly flat
- Handwriting naturally slopes
- Table surface not level

A strict horizontal line (0°) would miss many valid pairs. The tilt-search allows small deviations while preventing wild mismatches.

### Vertical Drift Prevention

Without `max_vertical_shift`, a long horizontal gap combined with even a small angle can drift the line into a completely different row:

```
Row 1:  [Coffee ________________] 
Row 2:  [Tea _____] [3.50]
Row 3:  [Juice _______________] [4.00]
```

If "Coffee" has a long gap and we allow any tilt, even 2° over that distance could reach down to "4.00" (wrong row). The vertical drift limit caps this based on typical text height.

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
**Problem**: First-match-wins means a greedy item can "steal" a price that belongs to a later item.

**Example failure**:
```
Row 1: [Coffee _____________________]
Row 2: [Tea ___] [3.50]
```
If "Coffee" tilts down and collides with "3.50" first (within ±8°), it claims that price. "Tea" becomes an orphan even though "3.50" is visually closer to "Tea".

**Better solution**: Try all possible pairings, score by distance + angle, then solve as bipartite matching problem (Hungarian algorithm).

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

**Purpose**: The classical fallback exists purely for offline testing when PaddleOCR isn't available. It's **not a replacement** for a trained detector.

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
| **Pairing** | Tilted horizontal line (geometric) | Y-distance nearest neighbor |
| **Column Detection** | None | Automatic left/right split |
| **Price Extraction** | None | Regex + OCR error correction |
| **Currency Detection** | None | TND/EUR/€ with fallback logic |
| **Validation** | None | File size, format, dimensions |
| **Error Handling** | None | Custom exception hierarchy |
| **Logging** | Print statements | Structured logging |
| **Thread Safety** | Not applicable | Lock-protected singletons |

**Key difference**: The experiment focuses purely on the geometric pairing algorithm, while the production pipeline is a complete end-to-end system with validation, logging, error handling, and production-ready features.

## Future Improvements

### High Impact
1. **Column clustering filter**: Reject false-positive boxes outside established column bands
2. **Bipartite matching**: Replace greedy first-match-wins with optimal global assignment
3. **Multi-layout detection**: Auto-detect vertical vs horizontal layout, switch algorithms

### Medium Impact
4. **Confidence scoring**: Weight pairings by distance + angle, flag low-confidence pairs
5. **Semantic validation**: Use OCR text to confirm item-price pairs make sense (text on left, number on right)
6. **Multi-pass pairing**: Run algorithm with different `MAX_TILT_DEG` values, merge results

### Low Impact
7. **Better visualization**: Color-code by confidence, show rejected candidates
8. **Parameter auto-tuning**: Learn optimal `MAX_TILT_DEG` / `CATEGORY_HEIGHT_RATIO` per image
9. **Export results**: Save structured JSON output (not just visualization)

---

**Made with dedication by Hamza Z** ✨
