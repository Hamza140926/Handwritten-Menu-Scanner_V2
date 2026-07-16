"""
Generates a synthetic handwritten-menu training set using sjvasquez's
handwriting-synthesis repo, rasterizes each line to a PNG crop, augments
it toward "photo of paper" realism, and writes a manifest.csv compatible
with the training/ scripts from before (train_trocr.py, evaluate.py).

Two deliberate differences from a naive "render the whole menu" approach:

1. ONE FIELD PER CROP, not one line per menu row.
   Your detection.py finds each name and each price as SEPARATE regions
   at inference time (pipeline.py's docstring documents this, confirmed
   against a real scan) - never one "Espresso ..... 2.50" string. This
   script generates one crop per name, one per price, one per category
   header, so training data matches production input shape.

2. STYLE/BIAS DIVERSITY, tracked per sample.
   Every crop's style id and bias are recorded in the manifest so the
   analysis notebook and the train/val/test split can be STYLE-AWARE -
   see split logic below and the big caveat in README_SYNTHETIC.md.

Usage:
    python generate_synthetic_dataset.py \\
        --out_dir dataset_synth/ \\
        --repo_dir ../handwriting-synthesis \\
        --samples_per_field 3

Produces:
    dataset_synth/crops/*.png
    dataset_synth/manifest.csv   columns: crop_path,text,split,field_type,style_id,bias,category

Split strategy (see README_SYNTHETIC.md for why this matters):
    - style-disjoint (enforced): each style id is assigned to exactly
      one of train/val/test, so no split sees a handwriting identity
      used in another split. This is the split that's actually applied.
    - vocab overlap (reported, not enforced): each sample's item name
      is separately tagged with a "vocab_bucket" - the analysis notebook
      uses this to report what fraction of val/test text also shows up
      in training text, so you know how much of any accuracy gain is
      "new style, same words" vs "genuinely unseen text" - without
      shrinking val/test to be unusably small by requiring both axes
      held out simultaneously.
"""
import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import cv2
import unicodedata

from vocabulary import get_menu_items, format_price
from augment import photo_realism_pipeline

FIELDS = ["crop_path", "text", "split", "field_type", "style_id", "bias", "category", "vocab_bucket"]

# Valid character set from the handwriting synthesis model
VALID_CHARS = {'m', 'A', 'c', '9', ' ', 'y', 'j', '?', 'F', 'D', '6', 'J', 'K', ')', 
               'f', '2', '.', 'L', '4', 'U', 'W', 's', 'w', '(', '8', 'O', 'g', 'C', 
               "'", 'z', '"', '!', 'I', 't', 'M', ',', 'P', 'a', 'q', 'r', '#', 'h', 
               'T', '3', 'b', '7', 'H', 'N', 'V', 'p', ';', 'x', '1', 'l', 'o', 'R', 
               'e', '\x00', 'B', '5', 'v', 'n', '0', ':', 'E', 'k', 'u', 'G', 'i', 
               'Y', '-', 'S', 'd'}


def sanitize_text(text: str) -> str:
    """Remove or replace characters not supported by the handwriting model.
    
    Strategy:
    1. Try to decompose accented characters (é -> e, è -> e)
    2. Replace common special characters with supported alternatives
    3. Remove any remaining unsupported characters
    """
    # First, try to normalize accented characters (NFD = decomposed form)
    # This turns 'é' into 'e' + accent, then we can strip the accent
    normalized = unicodedata.normalize('NFD', text)
    # Keep only ASCII characters after decomposition
    ascii_text = ''.join(c for c in normalized if ord(c) < 128)
    
    # Replace common special characters
    replacements = {
        '€': '',  # Remove euro symbol
        '&': 'and',  # Replace ampersand
        'Q': 'q',  # Lowercase Q (not in valid set but q is)
    }
    
    for old, new in replacements.items():
        ascii_text = ascii_text.replace(old, new)
    
    # Filter to only valid characters
    sanitized = ''.join(c for c in ascii_text if c in VALID_CHARS)
    
    return sanitized.strip()

# Valid style ids in sjvasquez/handwriting-synthesis (13 predefined styles).
ALL_STYLE_IDS = list(range(13))


def assign_style_splits(style_ids, val_fraction=0.2, test_fraction=0.2, seed=7):
    rng = random.Random(seed)
    shuffled = style_ids[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))
    test_styles = set(shuffled[:n_test])
    val_styles = set(shuffled[n_test:n_test + n_val])
    train_styles = set(shuffled[n_test + n_val:])
    return train_styles, val_styles, test_styles


def assign_vocab_buckets(items, val_fraction=0.15, test_fraction=0.15, seed=11):
    """Assigns each (category, name) item identity to train/val/test as
    a DIAGNOSTIC label only (see pick_split) - not used to force the
    actual split, just recorded so the analysis notebook can report how
    much a val/test sample's exact text also appears in training text.

    Earlier version of this script used both style AND vocab as hard,
    simultaneously-required split constraints - that starved val/test
    down to ~2% of samples each (verified: 8/8 out of 364 in a test
    run), which is too small to trust as a held-out set. Style alone is
    the primary generalization axis here (a new handwriting identity is
    a much bigger domain shift than a new item name for a character-
    level reader), so it drives the actual split; vocab overlap is
    reported, not enforced.
    """
    rng = random.Random(seed)
    names = sorted({name for _, name, _ in items})
    rng.shuffle(names)
    n = len(names)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))
    test_names = set(names[:n_test])
    val_names = set(names[n_test:n_test + n_val])
    train_names = set(names[n_test + n_val:])
    return train_names, val_names, test_names


def pick_split(style_id, train_styles, val_styles, test_styles):
    """Style-disjoint split: every sample rendered in a given style id
    always lands in the same split. This is the enforced constraint -
    it guarantees no handwriting identity leaks across train/val/test."""
    if style_id in test_styles:
        return "test"
    if style_id in val_styles:
        return "val"
    return "train"


def vocab_bucket_for(name, train_names, val_names, test_names):
    if name in test_names:
        return "test"
    if name in val_names:
        return "val"
    return "train"


def render_line(hand, text: str, style_id: int, bias: float, svg_path: Path):
    hand.write(
        filename=str(svg_path),
        lines=[text],
        biases=[bias],
        styles=[style_id],
        stroke_widths=[1.2],
    )


def svg_to_png(svg_path: Path, png_path: Path, scale: float = 4.0):
    import cairosvg
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), scale=scale, background_color="white")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--repo_dir", required=True, help="Path to cloned sjvasquez/handwriting-synthesis")
    parser.add_argument("--samples_per_field", type=int, default=2,
                         help="How many (style, bias) renderings to generate per name/price/header string")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Save current directory and change to repo directory so Hand class can find checkpoints
    original_dir = os.getcwd()
    repo_dir = Path(args.repo_dir).resolve()
    os.chdir(repo_dir)
    
    sys.path.insert(0, str(repo_dir))
    from demo import Hand  # noqa: E402  (must import after sys.path insert)

    rng = random.Random(args.seed)
    # Use absolute paths for output directories since we changed working directory
    out_dir = Path(original_dir) / args.out_dir
    crops_dir = out_dir / "crops"
    tmp_svg_dir = out_dir / "_tmp_svg"
    crops_dir.mkdir(parents=True, exist_ok=True)
    tmp_svg_dir.mkdir(parents=True, exist_ok=True)

    items = get_menu_items(seed=args.seed)
    train_styles, val_styles, test_styles = assign_style_splits(ALL_STYLE_IDS)
    train_names, val_names, test_names = assign_vocab_buckets(items)

    print(f"Style split (enforced) -> train:{sorted(train_styles)} val:{sorted(val_styles)} test:{sorted(test_styles)}")
    print(f"Vocab bucket (diagnostic only) -> train:{len(train_names)} val:{len(val_names)} test:{len(test_names)} item names")

    hand = Hand()
    rows = []
    counter = 0

    # Build the flat list of (field_type, text, category, item_name) to render.
    # item_name is used only for vocab-split lookup - category headers use
    # the category itself as the "name" key.
    fields_to_render = []
    seen_categories = set()
    for category, name, price in items:
        if category not in seen_categories:
            fields_to_render.append(("header", category.upper(), category, category))
            seen_categories.add(category)
        fields_to_render.append(("name", name, category, name))
        price_text = format_price(price, rng)
        fields_to_render.append(("price", price_text, category, name))

    total = len(fields_to_render) * args.samples_per_field
    print(f"Rendering {total} crops ({len(fields_to_render)} fields x {args.samples_per_field} samples each)...")

    for field_type, text, category, vocab_key in fields_to_render:
        for _ in range(args.samples_per_field):
            style_id = rng.choice(ALL_STYLE_IDS)
            bias = round(rng.uniform(0.4, 0.95), 2)
            
            # Sanitize text to remove unsupported characters
            sanitized_text = sanitize_text(text)
            if not sanitized_text:
                print(f"  skipping {text!r} - no valid characters after sanitization")
                continue

            svg_path = tmp_svg_dir / f"sample_{counter:05d}.svg"
            try:
                render_line(hand, sanitized_text, style_id, bias, svg_path)
            except Exception as e:
                print(f"  render failed for {text!r} -> {sanitized_text!r} (style={style_id}): {e}")
                continue

            png_raw = tmp_svg_dir / f"sample_{counter:05d}.png"
            try:
                svg_to_png(svg_path, png_raw)
            except Exception as e:
                print(f"  rasterize failed for {text!r}: {e}")
                continue

            image = cv2.imread(str(png_raw))
            if image is None:
                print(f"  could not load rasterized PNG for {text!r}, skipping")
                continue

            augmented = photo_realism_pipeline(image, rng)
            if augmented.size == 0:
                print(f"  blank render for {text!r}, skipping")
                continue

            crop_name = f"{field_type}_{counter:05d}.png"
            crop_path = crops_dir / crop_name
            cv2.imwrite(str(crop_path), augmented)

            split = pick_split(style_id, train_styles, val_styles, test_styles)
            vocab_bucket = vocab_bucket_for(vocab_key, train_names, val_names, test_names)

            rows.append({
                "crop_path": str(Path("crops") / crop_name),
                "text": sanitized_text,  # Store sanitized text in manifest
                "split": split,
                "field_type": field_type,
                "style_id": style_id,
                "bias": bias,
                "category": category,
                "vocab_bucket": vocab_bucket,
            })
            counter += 1
            if counter % 50 == 0:
                print(f"  {counter}/{total} rendered")

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    split_counts = {}
    for r in rows:
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1

    print(f"\nDone. {len(rows)} crops written to {manifest_path}")
    print(f"Split sizes: {split_counts}")
    print("Next: open dataset_analysis.ipynb to check for leakage and dataset quality before training.")


if __name__ == "__main__":
    main()
