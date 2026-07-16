# Training your own handwriting recognition model

Your pipeline currently uses `microsoft/trocr-base-handwritten` completely
unmodified (see `recognition.py`'s docstring: "Pretrained weights only, no
fine-tuning yet"). This is about fine-tuning that same model on *your*
menus' handwriting and vocabulary, not training something from scratch.
Training from scratch would need tens of thousands of labeled examples;
fine-tuning a strong pretrained baseline can show real gains from a few
hundred.

## Quick Start: Training on Synthetic Dataset

If you've already generated synthetic data using `dataset_generator/synthetic/`, you can start training immediately:

```bash
cd training
python train_synthetic.py \
    --dataset_dir ../dataset_generator/synthetic/dataset_synth \
    --output_model_dir ../models/trocr_menu_synth_v1 \
    --eval_test
```

This uses the pre-generated synthetic dataset (~4400 samples) with proper train/val/test splits based on handwriting styles. See "Synthetic Dataset Training" section below for details.

Detection (PaddleOCR, finds *where* text is) is left alone here. It's a
generic, well-trained model for finding text regions on a page, and
region-finding usually isn't where accuracy problems come from with
handwritten menus - reading the handwriting is. Only revisit detection
fine-tuning if `detect_text_regions` is missing/misplacing whole regions
in your real photos, which is a different (heavier) project.

## Setup

```bash
pip install transformers torch torchvision pillow opencv-python evaluate jiwer
```

(`evaluate` + `jiwer` are only needed for `train_trocr.py`/`evaluate.py`
- they compute Character/Word Error Rate.)

Place this `training/` folder next to your pipeline's `src/` folder:

```
your-project/
  src/           <- your existing pipeline (preprocessing.py, detection.py, ...)
  training/      <- these 4 scripts
```

## Workflow

### 1. Collect raw photos

Take photos of real menus the way your users actually will: same phone
types, same lighting variety, same handwriting styles. 15-30 source
menu photos is a reasonable starting point - each one yields many
individual crops (one per detected line), so this goes further than it
sounds.

### 2. Extract crops

```bash
cd training
python extract_crops.py --images_dir raw_menus/ --out_dir dataset/ --src_dir ../src
```

This runs your *actual* `preprocess_image` + `detect_text_regions` over
each photo and saves every detected region as its own small image in
`dataset/crops/`, plus `dataset/manifest.csv` (empty `text` column, to
fill in next). Re-running with new photos only adds new crops - it won't
touch labels you've already entered.

### 3. Label

```bash
python label_tool.py --out_dir dataset/
```

Opens each unlabeled crop in a window; type what it actually says in the
terminal and hit Enter. Saves after every single entry, so it's safe to
label 20 crops, stop, and resume days later. Use `header` for category
headers (e.g. "Coffee") and `junk` for unusable crops (bad detections,
blank boxes) - both get excluded from training automatically.

**How much to label:** aim for at least a few hundred non-junk labels
before your first training run. Below ~100, `train_trocr.py` will still
run and warn you, but treat the resulting CER as a rough signal, not a
reliable number - the validation set is too small to trust.

### 4. Fine-tune

```bash
python train_trocr.py --out_dir dataset/ --output_model_dir ../models/trocr_menu_v1
```

Fine-tunes from `microsoft/trocr-base-handwritten`, evaluates Character
Error Rate (CER) each epoch, keeps the best checkpoint, and stops early
if validation CER stops improving. On your own GPU this should take
anywhere from a few minutes (small dataset) to an hour or two (1000+
examples, many epochs).

### 5. Evaluate before/after

```bash
python evaluate.py --out_dir dataset/ --model microsoft/trocr-base-handwritten
python evaluate.py --out_dir dataset/ --model ../models/trocr_menu_v1
```

Run both and compare CER/WER on the same held-out `val` rows. Only swap
the model into production if the fine-tuned number is actually better -
with a small dataset it's entirely possible for fine-tuning to overfit
and get *worse* on held-out data, which is exactly what this step is for.

### 6. Plug it into your pipeline

Your `recognition.py` already loads the model via
`TrOCRProcessor.from_pretrained(cfg.model_checkpoint)`, where
`cfg.model_checkpoint` comes from `RecognitionConfig.model_checkpoint`
in `config.py`. Since `from_pretrained` accepts a local directory just
as happily as a Hugging Face Hub name, this is a one-line change - no
code changes needed in `recognition.py` itself:

```python
# src/config.py
@dataclass
class RecognitionConfig:
    model_checkpoint: str = "/absolute/path/to/models/trocr_menu_v1"
```

Or, without editing code, save a config JSON with
`Config().save(path)` / `load_config(path)` (already supported in
`config.py`) pointing `recognition.model_checkpoint` at your fine-tuned
model directory.

## A note on iterating

This is meant to be a loop, not a one-shot: after step 6, run the full
pipeline on new real menu photos, look at which items come out wrong,
add those specific crops (and similar-looking ones) to your labeled set,
and re-run steps 4-5. A model fine-tuned once on an initial batch and a
model that's gone through 2-3 rounds of "label the actual failures" tend
to look very different in practice.

---

## Synthetic Dataset Training

### Overview

The `dataset_generator/synthetic/` directory contains tools to generate a large synthetic handwritten menu dataset using [sjvasquez/handwriting-synthesis](https://github.com/sjvasquez/handwriting-synthesis). This provides an initial training dataset without manual labeling.

**Generated dataset stats:**
- ~4,400 samples total
- ~3,000 train / ~650 val / ~670 test
- Field types: menu headers, item names, prices
- 13 different handwriting styles
- Style-disjoint splits (no handwriting leak between splits)

### Training on Synthetic Data

```bash
cd training
python train_synthetic.py \
    --dataset_dir ../dataset_generator/synthetic/dataset_synth \
    --output_model_dir ../models/trocr_menu_synth_v1 \
    --batch_size 16 \
    --max_epochs 20 \
    --lr 5e-5 \
    --eval_test
```

**Key arguments:**
- `--dataset_dir`: Path to synthetic dataset (contains manifest.csv and crops/)
- `--output_model_dir`: Where to save the fine-tuned model
- `--batch_size`: Training batch size (increase if you have GPU memory)
- `--max_epochs`: Maximum training epochs
- `--lr`: Learning rate
- `--eval_test`: Evaluate on test set after training

**Expected training time:**
- With GPU (e.g., RTX 3060): 15-30 minutes
- With CPU: 3-6 hours (not recommended)

### Important Caveats

**Synthetic vs Real Data:**
The synthetic dataset is generated from a handwriting synthesis model, not real photographed menus. This means:

1. ✓ Good for: Learning character-level recognition patterns
2. ✓ Good for: Getting initial model weights better than base TrOCR
3. ✗ Limited: Real handwriting has more variation than synthetic
4. ✗ Limited: Real photos have lighting/angle/paper texture issues

**What to do:**
1. Train on synthetic data first (quick baseline)
2. Test on a small set of real menu photos (20-30 crops)
3. If real-world CER is much higher, collect and label real data
4. Fine-tune again on mixed synthetic + real data

### Using the Fine-tuned Model

After training, update your pipeline configuration:

```python
# src/config.py
@dataclass
class RecognitionConfig:
    model_checkpoint: str = r"C:\path\to\models\trocr_menu_synth_v1"
```

Or programmatically:
```python
from src.config import Config
config = Config()
config.recognition.model_checkpoint = r"C:\path\to\models\trocr_menu_synth_v1"
```

### Mixing Synthetic and Real Data

For best results, combine synthetic data with real labeled data:

1. Generate synthetic dataset (already done)
2. Collect and label 200-500 real menu crops using `extract_crops.py` + `label_tool.py`
3. Merge manifests:
```python
import pandas as pd

synth = pd.read_csv("dataset_generator/synthetic/dataset_synth/manifest.csv")
real = pd.read_csv("training/dataset/manifest.csv")

# Add any missing columns
for col in synth.columns:
    if col not in real.columns:
        real[col] = "unknown"

combined = pd.concat([synth, real], ignore_index=True)
combined.to_csv("training/dataset_mixed/manifest.csv", index=False)
```

4. Train on combined dataset using `train_trocr.py`

---
