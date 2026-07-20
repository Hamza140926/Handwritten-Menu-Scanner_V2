"""
Generate synthetic handwritten NUMBERS dataset for fine-tuning number recognition.

This script focuses exclusively on price/number recognition, generating crops
of handwritten numbers in various formats. Use this to continue training from
an existing checkpoint to improve number recognition accuracy.

Output structure matches the main synthetic dataset for compatibility with
train_synthetic.py.

Usage:
    python generate_numbers_dataset.py \\
        --out_dir dataset_numbers/ \\
        --repo_dir ../handwriting-synthesis-master \\
        --samples_per_price 10
        
Then continue training from checkpoint:
    python ../../training/train_synthetic.py \\
        --base_checkpoint ../../models/trocr_menu_v1_epoch2 \\
        --dataset_dir dataset_numbers/ \\
        --output_model_dir ../../models/trocr_menu_v2_numbers \\
        --max_epochs 3 \\
        --lr 1e-5
"""
import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import cv2

# Import from existing modules
from augment import photo_realism_pipeline

# Valid character set from handwriting synthesis model (for numbers: 0-9, ., ,)
VALID_CHARS = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', ',', ' '}


FIELDS = ["crop_path", "text", "split", "field_type", "style_id", "bias"]

ALL_STYLE_IDS = list(range(13))


def generate_number_variations(seed=42):
    """
    Generate clean number/price formats (0-99, no currency symbols).
    
    Includes:
    - Integers: "0", "1", "2", ..., "99"
    - Two decimals: "5.50", "12.00", "8.75"
    - Three decimals: "5.500", "12.000", "8.750"
    - Comma separator: "5,50", "12,00"
    - Common fractions: "0.5", "1.5", "2.5", etc.
    
    Total: ~200 unique numbers
    With 5 samples each = ~1000 total samples
    """
    rng = random.Random(seed)
    numbers = set()
    
    # Integers 0-99
    for i in range(100):
        numbers.add(str(i))
    
    # Common price points with 2 decimals
    common_prices = [
        0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
        5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
        10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0,
        16.0, 17.0, 18.0, 19.0, 20.0, 22.0, 25.0, 28.0, 30.0, 35.0,
        40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0,
        90.0, 95.0, 99.0
    ]
    
    for price in common_prices:
        # 2 decimals with dot
        numbers.add(f"{price:.2f}")
        
        # 3 decimals (TND style)
        numbers.add(f"{price:.3f}")
        
        # Comma separator (EU style)
        numbers.add(f"{price:.2f}".replace(".", ","))
    
    return sorted(numbers)


def sanitize_text(text: str) -> str:
    """Keep only characters supported by handwriting synthesis model."""
    return ''.join(c for c in text if c in VALID_CHARS).strip()


def assign_style_splits(style_ids, val_fraction=0.15, test_fraction=0.15, seed=7):
    """Assign handwriting styles to train/val/test splits (style-disjoint)."""
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


def pick_split(style_id, train_styles, val_styles, test_styles):
    """Determine split based on style_id."""
    if style_id in test_styles:
        return "test"
    if style_id in val_styles:
        return "val"
    return "train"


def render_line(hand, text: str, style_id: int, bias: float, svg_path: Path):
    """Render a line of text using the Hand class."""
    hand.write(
        filename=str(svg_path),
        lines=[text],
        biases=[bias],
        styles=[style_id],
        stroke_widths=[1.2],
    )


def svg_to_png(svg_path: Path, png_path: Path, scale: float = 4.0):
    """Convert SVG to PNG using cairosvg."""
    import cairosvg
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), scale=scale, background_color="white")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out_dir", required=True, help="Output directory for dataset")
    parser.add_argument("--repo_dir", required=True, help="Path to handwriting-synthesis-master repo")
    parser.add_argument("--samples_per_price", type=int, default=5,
                       help="How many style/bias variations to generate per unique price")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Save current directory and change to repo directory so Hand class can find checkpoints
    original_dir = os.getcwd()
    repo_dir = Path(args.repo_dir).resolve()
    
    if not (repo_dir / "demo.py").exists():
        print(f"ERROR: {repo_dir}/demo.py not found")
        print("Make sure --repo_dir points to the handwriting-synthesis-master folder")
        sys.exit(1)
    
    os.chdir(repo_dir)
    sys.path.insert(0, str(repo_dir))
    from demo import Hand  # noqa: E402  (must import after sys.path insert)
    
    print("=" * 70)
    print("NUMBERS DATASET GENERATOR")
    print("=" * 70)
    
    # Generate number variations
    print("\nGenerating number variations...")
    numbers = generate_number_variations(seed=args.seed)
    print(f"  Total unique numbers: {len(numbers)}")
    
    # Assign styles to splits
    train_styles, val_styles, test_styles = assign_style_splits(ALL_STYLE_IDS, seed=args.seed)
    print(f"\nStyle splits:")
    print(f"  Train styles: {sorted(train_styles)}")
    print(f"  Val styles:   {sorted(val_styles)}")
    print(f"  Test styles:  {sorted(test_styles)}")
    
    # Setup directories (use absolute paths since we changed working directory)
    rng = random.Random(args.seed)
    out_dir = Path(original_dir) / args.out_dir
    crops_dir = out_dir / "crops"
    tmp_svg_dir = out_dir / "_tmp_svg"
    crops_dir.mkdir(parents=True, exist_ok=True)
    tmp_svg_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Hand for rendering
    hand = Hand()
    
    # Generate samples
    print(f"\nGenerating {len(numbers) * args.samples_per_price} samples...")
    print("  (this may take a while...)")
    
    manifest_rows = []
    crop_idx = 0
    
    for num_text in numbers:
        sanitized = sanitize_text(num_text)
        if not sanitized:
            print(f"  Skipping '{num_text}' (no valid characters)")
            continue
        
        # Generate multiple variations with different styles/biases
        for _ in range(args.samples_per_price):
            style_id = rng.choice(ALL_STYLE_IDS)
            bias = rng.uniform(0.3, 1.0)  # Moderate to high bias for clearer numbers
            
            svg_path = tmp_svg_dir / f"price_{crop_idx:05d}.svg"
            png_raw = tmp_svg_dir / f"price_{crop_idx:05d}.png"
            
            # Render to SVG
            try:
                render_line(hand, sanitized, style_id, bias, svg_path)
            except Exception as e:
                print(f"  Render failed for '{sanitized}': {e}")
                continue
            
            # Convert SVG to PNG
            try:
                svg_to_png(svg_path, png_raw)
            except Exception as e:
                print(f"  Rasterize failed for '{sanitized}': {e}")
                continue
            
            # Load PNG
            image = cv2.imread(str(png_raw))
            if image is None:
                print(f"  Could not load PNG for '{sanitized}', skipping")
                continue
            
            # Apply photo-realism augmentations
            try:
                img_augmented = photo_realism_pipeline(image, rng)
            except Exception as e:
                print(f"  Augmentation error for '{sanitized}': {e}")
                continue
            
            if img_augmented.size == 0:
                print(f"  Blank render for '{sanitized}', skipping")
                continue
            
            # Save crop
            crop_filename = f"price_{crop_idx:05d}.png"
            crop_path = crops_dir / crop_filename
            cv2.imwrite(str(crop_path), img_augmented)
            
            # Determine split
            split = pick_split(style_id, train_styles, val_styles, test_styles)
            
            # Add to manifest
            manifest_rows.append({
                "crop_path": f"crops/{crop_filename}",
                "text": sanitized,
                "split": split,
                "field_type": "price",
                "style_id": style_id,
                "bias": f"{bias:.3f}"
            })
            
            crop_idx += 1
            
            if crop_idx % 100 == 0:
                print(f"  Generated {crop_idx} samples...")
    
    # Write manifest
    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    
    # Stats
    train_count = sum(1 for r in manifest_rows if r["split"] == "train")
    val_count = sum(1 for r in manifest_rows if r["split"] == "val")
    test_count = sum(1 for r in manifest_rows if r["split"] == "test")
    
    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nTotal samples: {len(manifest_rows)}")
    print(f"  Train: {train_count}")
    print(f"  Val:   {val_count}")
    print(f"  Test:  {test_count}")
    print(f"\nOutput directory: {out_dir.resolve()}")
    print(f"Manifest: {manifest_path}")
    print(f"\nNext steps:")
    print(f"  1. Continue training from your existing checkpoint:")
    print(f"     python ../../training/train_synthetic.py \\")
    print(f"       --base_checkpoint ../../models/trocr_menu_v1_epoch2 \\")
    print(f"       --dataset_dir {args.out_dir} \\")
    print(f"       --output_model_dir ../../models/trocr_menu_v2_numbers \\")
    print(f"       --max_epochs 3 \\")
    print(f"       --lr 1e-5")
    print()


if __name__ == "__main__":
    main()
