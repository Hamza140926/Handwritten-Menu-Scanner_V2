# Synthetic Handwritten Menu Dataset Generator

This project generates synthetic handwritten menu item images for training OCR models. It uses the [handwriting-synthesis](https://github.com/sjvasquez/handwriting-synthesis) model to create realistic handwritten text samples with various styles and augmentations.

## Project Overview

The goal is to create a diverse synthetic dataset of handwritten menu text (item names, prices, category headers) that can be used to fine-tune optical character recognition (OCR) models like TrOCR for reading real handwritten menus.

### Why Synthetic Data?

- **Real handwritten menus are hard to collect** - labor-intensive to photograph, transcribe, and annotate
- **Synthetic data provides volume** - can generate thousands of samples programmatically
- **Style diversity** - the handwriting-synthesis model provides 13 different handwriting styles
- **Cost-effective** - no manual data collection or annotation needed
- **Foundation for fine-tuning** - pre-trained OCR models need adaptation to handwritten text

### Important Caveat

⚠️ **Synthetic data is NOT a replacement for real data.** Models trained on this synthetic dataset learn to read the handwriting-synthesis model's output, not real human handwriting. Always validate on real photographed menus before deploying to production.

---

## Project Structure

```
sidequest/
├── handwriting-synthesis-master/    # Cloned handwriting synthesis model
│   ├── checkpoints/                 # Pre-trained model weights
│   ├── styles/                      # Style encoding files (13 styles)
│   ├── demo.py                      # Core handwriting generation API
│   └── ...
│
└── synthetic/                       # Our dataset generation scripts
    ├── generate_synthetic_dataset.py   # Main menu dataset generation
    ├── generate_numbers_dataset.py     # Numbers-only dataset generation (NEW)
    ├── clean_dataset.py                # Interactive dataset cleaning tool (NEW)
    ├── vocabulary.py                   # Menu items and price formatting
    ├── augment.py                      # Photo-realism augmentation
    ├── dataset_analysis.ipynb          # Quality checks for main dataset
    ├── dataset_numbers_analysis.ipynb  # Quality checks for numbers dataset (NEW)
    ├── README_SYNTHETIC.md             # Detailed usage instructions
    ├── dataset_synth/                  # Generated menu dataset
    │   ├── crops/                      # PNG images of handwritten text
    │   ├── manifest.csv                # Labels and metadata
    │   └── _tmp_svg/                   # Temporary SVG files
    └── dataset_numbers/                # Generated numbers dataset (NEW)
        ├── crops/                      # PNG images of handwritten numbers
        ├── manifest.csv                # Labels and metadata
        └── _tmp_svg/                   # Temporary SVG files
```

---

## Setup Instructions

### 1. Clone the Handwriting Synthesis Repository

The handwriting-synthesis model is the core engine that generates handwritten text:

```bash
cd C:\Users\zussl\Desktop\sidequest
git clone https://github.com/sjvasquez/handwriting-synthesis.git handwriting-synthesis-master
```

**What it provides:**
- Pre-trained RNN model for generating handwritten strokes
- 13 distinct handwriting style identities
- Checkpoint files with learned parameters
- API for generating SVG handwriting from text

### 2. Create a Locked Python Environment

We use a **locked environment** to ensure reproducibility and avoid dependency conflicts:

```bash
# Create a new conda environment
conda create -n handwriting python=3.8
conda activate handwriting

# Install dependencies
pip install tensorflow==1.15.0
pip install svgwrite numpy pandas matplotlib jupyter opencv-python cairosvg
```

**Why lock the environment?**
- **TensorFlow 1.15 compatibility** - The handwriting-synthesis model was built with TF 1.x (deprecated in 2021)
- **Reproducibility** - Same package versions = same results
- **Isolation** - Prevents conflicts with other Python projects
- **Stability** - Newer TF versions break backward compatibility

**Why TensorFlow 1.15 specifically?**
- The handwriting-synthesis checkpoint files use TF 1.x format
- TF 2.x removed many APIs the model relies on
- Migrating would require retraining (weeks of GPU time)

### 3. Install Additional Dependencies

```bash
pip install cairosvg opencv-python
```

**Why these packages?**
- **cairosvg** - Converts SVG (vector) to PNG (raster) images
- **opencv-python** - Image augmentation (noise, rotation, compression)
- **pandas** - CSV manifest generation and analysis
- **matplotlib** - Visualization in analysis notebook

---

## How It Works

### Generation Pipeline

```
1. Menu Items (vocabulary.py)
   └─> Category/name/price tuples
   
2. Text Sanitization (generate_synthetic_dataset.py)
   └─> Remove unsupported chars (é→e, €→removed, &→and)
   
3. Handwriting Rendering (handwriting-synthesis)
   └─> Generate SVG strokes for each text
   └─> Apply random style (0-12) and bias (0.4-0.95)
   
4. Rasterization (cairosvg)
   └─> Convert SVG → PNG at 4x scale
   
5. Augmentation (augment.py)
   └─> Add noise, rotation, blur, compression
   └─> Simulate "photo of paper" realism
   
6. Output
   └─> crops/: PNG images
   └─> manifest.csv: Labels + metadata
```

### Style-Disjoint Split Strategy

**Critical design decision:** Each of the 13 handwriting styles is assigned exclusively to train, val, or test:

- **Train styles**: [0, 1, 2, 4, 5, 6, 8, 9, 11] (9 styles)
- **Val styles**: [7, 12] (2 styles)
- **Test styles**: [3, 10] (2 styles)

**Why style-disjoint?**
- Tests model's ability to generalize to **new handwriting identities**
- Prevents memorization of specific stroke patterns
- More realistic evaluation (real deployment sees new handwriters)

**What about vocabulary overlap?**
- Text content (item names) intentionally overlaps ~96% between splits
- Enforcing both style AND vocabulary disjointness would shrink val/test to ~2%
- Priority: style generalization > vocabulary generalization (for character-level OCR)

---

## Usage

### Generate Full Menu Dataset

**Basic usage (small dataset for testing):**
```bash
cd C:\Users\zussl\Desktop\sidequest\synthetic
conda activate handwriting

python generate_synthetic_dataset.py \
    --out_dir dataset_synth/ \
    --repo_dir ..\handwriting-synthesis-master \
    --samples_per_field 3
```

**Recommended for training (larger dataset):**
```bash
python generate_synthetic_dataset.py \
    --out_dir dataset_synth/ \
    --repo_dir ..\handwriting-synthesis-master \
    --samples_per_field 10
```

**For production-quality dataset:**
```bash
python generate_synthetic_dataset.py \
    --out_dir dataset_synth/ \
    --repo_dir ..\handwriting-synthesis-master \
    --samples_per_field 20
```

**Parameters:**
- `--out_dir`: Output directory for crops and manifest
- `--repo_dir`: Path to handwriting-synthesis-master repository
- `--samples_per_field`: How many style/bias variations per text string
- `--seed`: Random seed for reproducibility (default: 0)

**Expected output:**
- 191 unique field strings (names, prices, headers)
- × `samples_per_field` = total crops
- Example: `--samples_per_field 10` → ~1,910 total images

---

### Generate Numbers Dataset (NEW)

For fine-tuning number/price recognition specifically, use the numbers dataset generator:

**Generate numbers dataset:**
```bash
cd C:\Users\zussl\Desktop\sidequest\synthetic
conda activate handwriting

python generate_numbers_dataset.py \
    --out_dir dataset_numbers/ \
    --repo_dir ..\handwriting-synthesis-master \
    --samples_per_price 5
```

**What it generates:**
- Numbers 0-99 (no currency symbols)
- Multiple formats:
  - Integers: `0`, `1`, `2`, ..., `99`
  - 2 decimals: `5.50`, `12.00`, `8.75`
  - 3 decimals: `5.500`, `12.000` (TND style)
  - Comma separator: `5,50`, `12,00` (EU style)
- ~200 unique numbers × 5 samples = ~1,000 total crops

**Parameters:**
- `--out_dir`: Output directory for numbers dataset
- `--repo_dir`: Path to handwriting-synthesis-master repository
- `--samples_per_price`: How many style/bias variations per number (default: 5)
- `--seed`: Random seed for reproducibility (default: 42)

**Use case:** Continue training from an existing checkpoint to improve number recognition accuracy:
```bash
python ../../training/train_synthetic.py \
    --base_checkpoint ../../models/trocr_menu_v1_epoch2 \
    --dataset_dir dataset_numbers/ \
    --output_model_dir ../../models/trocr_menu_v2_numbers \
    --max_epochs 3 \
    --lr 1e-5
```

---

### Clean Dataset (NEW)

After generating a dataset, use the interactive cleaning tool to review and remove invalid samples:

**Run the cleaning tool:**
```bash
cd C:\Users\zussl\Desktop\sidequest\synthetic
conda activate handwriting

python clean_dataset.py --dataset_dir dataset_numbers
```

**Controls:**
- **ENTER** or **SPACE**: Mark as valid (keep sample)
- **X**: Mark as invalid (delete sample)
- **S**: Skip (no decision)
- **B**: Go back to previous sample
- **Q** or **ESC**: Quit and process results

**What it does:**
1. Displays each image with its label and metadata
2. Lets you review quality one-by-one
3. Marks invalid samples for deletion
4. Deletes invalid images and updates manifest.csv
5. Backs up original manifest before modifications

**When to use:**
- After generating a dataset for the first time
- If you notice quality issues during training
- Before fine-tuning to ensure data quality

### Analyze Dataset Quality

**Before training**, always run the analysis notebook:

**For full menu dataset:**
```bash
jupyter notebook dataset_analysis.ipynb
```

**For numbers dataset:**
```bash
jupyter notebook dataset_numbers_analysis.ipynb
```

Both notebooks check:
1. ✅ **Style leakage** - Should be ZERO overlap (critical)
2. ✅ **Format distribution** - Variety in number/text formats
3. ✅ **Number/text coverage** - What appears in each split
4. ✅ **Image quality** - No blank/corrupted images
5. ✅ **Diversity** - Style distribution, bias range
6. ✅ **Visual samples** - Grid of rendered crops

**Go/No-Go decision:** If all checks pass, proceed to training. If style leakage detected, regenerate dataset.

---

## Key Design Decisions

### 1. **Per-Field Crops (Not Full Lines)**

❌ **Don't generate**: `"Espresso .............. 2.50"`

✅ **Do generate separately**:
- Crop 1: `"Espresso"` (name)
- Crop 2: `"2.50"` (price)

**Why?**
- The detection pipeline (`detection.py`) finds names and prices as **separate regions**
- Training on combined lines would teach a format the model never sees at inference
- Matches real-world input shape

### 2. **Text Sanitization**

The handwriting-synthesis model only supports basic ASCII. We sanitize:
- `é, è, û` → `e, e, u` (decompose accents)
- `€` → removed (currency symbol)
- `&` → `and`
- `Q` → `q` (uppercase Q not in model's charset)

**Tracked in manifest:** The sanitized text is stored, not the original.

### 3. **Photo-Realism Augmentation**

Raw SVG renders are too clean. We add:
- Gaussian noise
- Slight rotation (±2°)
- Gaussian blur
- JPEG compression artifacts
- Random brightness/contrast

**Goal:** Narrow the synthetic→real domain gap (doesn't eliminate it).

### 4. **Diverse Price Formats**

Simulates real menu handwriting variety:
- `4.50` (dot, 2 decimals)
- `4,50` (comma, 2 decimals)
- `12.500` (dot, 3 decimals - TND style)
- `5` (integer)
- `4.50 €` (with currency, 25% of time)

---

## Dataset Size Guidelines

### Full Menu Dataset

| Samples/Field | Total Crops | Use Case |
|---------------|-------------|----------|
| 3 | ~573 | Pipeline testing, proof-of-concept |
| 5 | ~955 | Minimum for light fine-tuning |
| 10 | ~1,910 | Small but usable dataset |
| 20 | ~3,820 | Recommended for training |
| 50 | ~9,550 | Large, robust dataset |

### Numbers Dataset

| Samples/Price | Total Crops | Use Case |
|---------------|-------------|----------|
| 3 | ~600 | Quick testing |
| 5 | ~1,000 | Minimum for fine-tuning (recommended) |
| 10 | ~2,000 | Strong number recognition |
| 20 | ~4,000 | Production-quality |

**Rule of thumb:** More is better, but diminishing returns after ~5,000 samples for fine-tuning pre-trained models.

---

## Troubleshooting

### Issue: "Can't load save_path when it is None"

**Cause:** Script can't find checkpoint files.

**Fix:** Ensure you're running from the handwriting-synthesis-master directory (the script changes working directory automatically in the fixed version).

### Issue: "Invalid character X detected"

**Cause:** Text contains characters not in the model's vocabulary.

**Fix:** Already handled by `sanitize_text()` function in the fixed version. Unsupported characters are removed or replaced.

### Issue: TensorFlow GPU warnings

**Safe to ignore:** The model runs fine on CPU. GPU support requires CUDA 10.0 (very old, not worth the setup effort).

### Issue: Rendering is slow

**Expected:** ~50-100 crops/minute. For 2,000 samples, expect ~20-40 minutes.

**Can't be parallelized** (model loads session globally).

---

## Next Steps After Generation

1. **Clean dataset (optional)** - Run `clean_dataset.py` to remove invalid samples
2. **Run analysis notebook** - Validate quality with `dataset_analysis.ipynb` or `dataset_numbers_analysis.ipynb`
3. **Train OCR model** - Use TrOCR or similar (see `training/train_synthetic.py`)
4. **Evaluate on synthetic test set** - Get baseline metrics
5. **Evaluate on real photos** - The real validation (expect CER gap)
6. **Fine-tune for numbers** - If needed, generate numbers dataset and continue training
7. **Iterate** - Adjust augmentation, generate more samples, or collect real data

---

## Technical Notes

### Why We Don't Use TensorFlow 2.x

The handwriting-synthesis model uses TF 1.x APIs that are incompatible with TF 2.x:
- `tf.contrib` module (removed in TF 2.0)
- `tf.nn.rnn_cell.RNNCell` (deprecated, API changed)
- Checkpoint format differences
- Session-based execution model (TF 2.x uses eager execution)

**Migration effort:** Would require rewriting model code + retraining (not worth it for this use case).

### Character Set Limitation

The model only supports these characters:
```
A-Z, a-z, 0-9, space, and: . , ! ? ' " ; : # ( ) -
```

**Not supported:** `é è û ñ € £ & @ Q` and others.

**Workaround:** Text sanitization (see `sanitize_text()` function).

### Style IDs Explained

Each style (0-12) represents a different person's handwriting:
- Captured from real handwriters
- Encoded as numpy arrays in `styles/style-X-strokes.npy`
- Used as conditioning input to the RNN

**Bias parameter:** Controls writing style variation (0.0 = more uniform, 1.0 = more varied strokes).

---

## References

- **Handwriting Synthesis Model**: https://github.com/sjvasquez/handwriting-synthesis
- **Original Paper**: "Generating Sequences With Recurrent Neural Networks" by Alex Graves
- **TrOCR**: Microsoft's Transformer-based OCR model for fine-tuning

---

## License

- Handwriting-synthesis model: MIT License (see original repo)
- This generation code: [Your license choice]

---

## Authors

- Dataset generation scripts: [Hamza Slimani]
- Handwriting-synthesis model: Sean Vasquez

---

## FAQ

**Q: Can I use this for commercial projects?**
A: Check the handwriting-synthesis license. The model is MIT licensed, but validate on real data before production use.

**Q: How accurate will my OCR model be?**
A: Depends heavily on: (1) dataset size, (2) base model quality, (3) how well augmentation matches your real photos. Expect 5-20% CER on synthetic test, potentially much higher on real photos initially.

**Q: Should I generate more data or collect real data?**
A: Both. Start with synthetic for volume, then collect ~100-500 real samples for validation and fine-tuning the domain gap.

**Q: Can I add more menu items?**
A: Yes! Edit `vocabulary.py` → `CATEGORIES` dict. Add your items, then regenerate.

**Q: Why is test/val text overlapping with train?**
A: Intentional. See "Style-Disjoint Split Strategy" section. Priority is style generalization.

**Q: Should I train on the full menu dataset or numbers dataset first?**
A: Train on the full menu dataset first, then fine-tune with the numbers dataset if number recognition is weak.

**Q: How do I know if I need the numbers dataset?**
A: If your model performs well on names but struggles with prices/numbers, generate and train on the numbers dataset as a second fine-tuning stage.

**Q: Can I combine both datasets?**
A: Yes, you can merge the manifest.csv files and combine the crops directories, but training in stages (full menu → numbers) often works better.
