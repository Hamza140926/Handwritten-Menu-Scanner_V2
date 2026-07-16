# Synthetic dataset from handwriting-synthesis

## Setup

```bash
git clone https://github.com/sjvasquez/handwriting-synthesis.git
pip install cairosvg opencv-python numpy pandas matplotlib jupyter
```

Put this `synthetic/` folder next to your cloned `handwriting-synthesis/`
repo (or pass `--repo_dir` pointing at it).

## Usage

```bash
cd synthetic
python generate_synthetic_dataset.py \
    --out_dir dataset_synth/ \
    --repo_dir ../handwriting-synthesis \
    --samples_per_field 3
```

Then open `dataset_analysis.ipynb`, point `DATASET_DIR` at `dataset_synth/`,
and run it top to bottom **before** training anything. It checks for style
leakage, reports vocabulary overlap, flags out-of-vocabulary characters,
finds broken/blank images, and checks format diversity. Go through every
section - don't skip to the summary cell.

Once it looks clean, the output (`dataset_synth/manifest.csv` +
`dataset_synth/crops/`) uses the exact same schema as the earlier
`training/` scripts (`train_trocr.py`, `evaluate.py`), so you can point
those at it directly:

```bash
python ../training/train_trocr.py --out_dir dataset_synth/ --output_model_dir ../models/trocr_menu_v1
python ../training/evaluate.py --out_dir dataset_synth/ --model ../models/trocr_menu_v1
```

## What changed from your original snippet, and why

**Per-field crops, not combined lines.** Your pipeline's `pipeline.py`
documents that name and price arrive as two separate detected regions at
inference time - never one `"Espresso ..... 2.50"` string. Training on
combined lines would teach the model a text format it will never actually
receive as input. This script renders one crop per name, one per price,
one per category header.

**Style/bias diversity.** One fixed style/bias trains a model that reads
one synthetic handwriting identity well and nothing else. Styles (0-12)
and biases are sampled per crop and recorded in the manifest.

**Photo-realism augmentation.** Rendered SVGs are clean vectors; real
photos (what `preprocessing.py` is built to handle) have noise, uneven
lighting, slight rotation, and compression artifacts. `augment.py` adds
approximations of all of these. This narrows, but does not close, the
gap between synthetic and real input.

## The caveat that matters most

**Every crop in this dataset - train, val, and test - comes from the same
handwriting-synthesis model.** The split enforces one thing rigorously:
no *handwriting style* used in training appears in val/test
(`style_id`-disjoint, verified by the notebook). It does **not** enforce
vocabulary separation (the notebook will show real, often high, text
overlap between splits - that's reported honestly, not hidden), and it
cannot, by construction, tell you anything about *photographed* handwriting,
paper texture, real pen pressure variation, or real people's handwriting
idiosyncrasies.

Concretely: a low CER on this val/test set tells you the model generalizes
across handwriting-synthesis styles it hasn't seen. It does not tell you
the model will read an actual scanned menu. Both things can independently
be true or false. Before trusting this model in your pipeline, get even a
small set of real photographed menu crops (20-30 is enough for a sanity
check, not a rigorous eval) and run `evaluate.py` against those too - a
big gap between synthetic-val CER and real-photo CER is expected and is
the actual signal you're looking for, not a bug.
