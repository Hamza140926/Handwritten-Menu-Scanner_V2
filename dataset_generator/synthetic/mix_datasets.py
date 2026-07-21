"""
Mix numbers and synthetic menu datasets with custom proportions.

Creates a new combined dataset by:
1. Sampling from numbers dataset (default 70%)
2. Sampling from synthetic menu dataset (default 30%)
3. Copying images to new directory
4. Creating combined manifest.csv
5. Preserving split assignments (train/val/test)

Usage:
    python mix_datasets.py \
        --numbers_dir dataset_numbers \
        --synthetic_dir dataset_synth \
        --output_dir dataset_mixed \
        --numbers_ratio 0.7
        
After mixing, clean the dataset:
    python clean_dataset.py --dataset_dir dataset_mixed
"""
import argparse
import csv
import shutil
from pathlib import Path
from collections import defaultdict
import random


def load_manifest(manifest_path):
    """Load manifest CSV and return as list of dicts."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def sample_by_split(samples, ratio, seed=42):
    """
    Sample from dataset while preserving split proportions.
    
    Args:
        samples: List of sample dicts with 'split' field
        ratio: Fraction of samples to keep (0.0 to 1.0)
        seed: Random seed
    
    Returns:
        List of sampled dicts
    """
    rng = random.Random(seed)
    
    # Group by split
    by_split = defaultdict(list)
    for sample in samples:
        by_split[sample['split']].append(sample)
    
    # Sample from each split proportionally
    sampled = []
    for split, split_samples in by_split.items():
        k = max(1, int(len(split_samples) * ratio))
        sampled.extend(rng.sample(split_samples, k))
    
    return sampled


def mix_datasets(numbers_dir, synthetic_dir, output_dir, numbers_ratio=0.7, seed=42):
    """
    Mix two datasets with specified ratio.
    
    Args:
        numbers_dir: Path to numbers dataset
        synthetic_dir: Path to synthetic menu dataset
        output_dir: Path to output mixed dataset
        numbers_ratio: Fraction of numbers samples (0.0 to 1.0)
        seed: Random seed for reproducibility
    """
    numbers_dir = Path(numbers_dir)
    synthetic_dir = Path(synthetic_dir)
    output_dir = Path(output_dir)
    
    # Load manifests
    print("=" * 70)
    print("MIXING DATASETS")
    print("=" * 70)
    
    numbers_manifest = load_manifest(numbers_dir / "manifest.csv")
    synthetic_manifest = load_manifest(synthetic_dir / "manifest.csv")
    
    print(f"\nSource datasets:")
    print(f"  Numbers:   {len(numbers_manifest)} samples")
    print(f"  Synthetic: {len(synthetic_manifest)} samples")
    
    # Calculate target sizes
    # We want final mix to be numbers_ratio:synthetic_ratio
    # Start with all numbers samples, then scale synthetic accordingly
    total_numbers = len(numbers_manifest)
    synthetic_ratio = 1.0 - numbers_ratio
    
    # Calculate how many synthetic samples needed to achieve the ratio
    # If numbers_ratio = 0.7, synthetic_ratio = 0.3
    # numbers / (numbers + synthetic) = 0.7
    # numbers / 0.7 = numbers + synthetic
    # synthetic = numbers / 0.7 - numbers = numbers * (1/0.7 - 1)
    target_synthetic = int(total_numbers * (synthetic_ratio / numbers_ratio))
    
    # If we don't have enough synthetic samples, scale down numbers instead
    if target_synthetic > len(synthetic_manifest):
        print(f"\n⚠️  Not enough synthetic samples to achieve {numbers_ratio:.0%} ratio")
        print(f"   Adjusting to use all available synthetic samples...")
        total_synthetic = len(synthetic_manifest)
        total_numbers = int(total_synthetic * (numbers_ratio / synthetic_ratio))
        numbers_samples = sample_by_split(numbers_manifest, total_numbers / len(numbers_manifest), seed)
        synthetic_samples = synthetic_manifest
    else:
        numbers_samples = numbers_manifest  # Use all numbers
        synthetic_samples = sample_by_split(synthetic_manifest, target_synthetic / len(synthetic_manifest), seed)
    
    print(f"\nMixed dataset composition:")
    print(f"  Numbers:   {len(numbers_samples)} samples ({len(numbers_samples)/(len(numbers_samples)+len(synthetic_samples))*100:.1f}%)")
    print(f"  Synthetic: {len(synthetic_samples)} samples ({len(synthetic_samples)/(len(numbers_samples)+len(synthetic_samples))*100:.1f}%)")
    print(f"  Total:     {len(numbers_samples) + len(synthetic_samples)} samples")
    
    # Create output directory
    output_crops = output_dir / "crops"
    output_crops.mkdir(parents=True, exist_ok=True)
    
    # Copy images and build combined manifest
    print(f"\nCopying images to {output_dir}...")
    combined_manifest = []
    counter = 0
    
    # Process numbers samples
    for sample in numbers_samples:
        src_path = numbers_dir / sample['crop_path']
        if not src_path.exists():
            print(f"  ⚠️  Skipping missing file: {src_path}")
            continue
        
        # Create new filename
        ext = src_path.suffix
        new_filename = f"mixed_{counter:05d}{ext}"
        dst_path = output_crops / new_filename
        
        # Copy image
        shutil.copy2(src_path, dst_path)
        
        # Add to manifest with updated path
        new_sample = sample.copy()
        new_sample['crop_path'] = f"crops/{new_filename}"
        new_sample['source'] = 'numbers'
        combined_manifest.append(new_sample)
        
        counter += 1
        if counter % 100 == 0:
            print(f"  Copied {counter} images...")
    
    # Process synthetic samples
    for sample in synthetic_samples:
        src_path = synthetic_dir / sample['crop_path']
        if not src_path.exists():
            print(f"  ⚠️  Skipping missing file: {src_path}")
            continue
        
        # Create new filename
        ext = src_path.suffix
        new_filename = f"mixed_{counter:05d}{ext}"
        dst_path = output_crops / new_filename
        
        # Copy image
        shutil.copy2(src_path, dst_path)
        
        # Add to manifest with updated path
        new_sample = sample.copy()
        new_sample['crop_path'] = f"crops/{new_filename}"
        new_sample['source'] = 'synthetic'
        combined_manifest.append(new_sample)
        
        counter += 1
        if counter % 100 == 0:
            print(f"  Copied {counter} images...")
    
    # Shuffle combined manifest to mix sources
    rng = random.Random(seed)
    rng.shuffle(combined_manifest)
    
    # Determine manifest fields (union of both datasets)
    all_fields = set()
    for sample in combined_manifest:
        all_fields.update(sample.keys())
    
    # Ensure core fields are present and in order
    core_fields = ['crop_path', 'text', 'split', 'field_type', 'style_id', 'bias', 'source']
    other_fields = sorted(all_fields - set(core_fields))
    fieldnames = [f for f in core_fields if f in all_fields] + other_fields
    
    # Write combined manifest
    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_manifest)
    
    # Print split statistics
    print(f"\nSplit distribution:")
    split_counts = defaultdict(lambda: {'numbers': 0, 'synthetic': 0})
    for sample in combined_manifest:
        split_counts[sample['split']][sample['source']] += 1
    
    for split in ['train', 'val', 'test']:
        if split in split_counts:
            nums = split_counts[split]['numbers']
            synth = split_counts[split]['synthetic']
            total = nums + synth
            print(f"  {split.capitalize():5s}: {total:4d} total ({nums:4d} numbers, {synth:4d} synthetic)")
    
    print("\n" + "=" * 70)
    print("MIXING COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {output_dir.resolve()}")
    print(f"Manifest: {manifest_path}")
    print(f"\nNext steps:")
    print(f"  1. Review dataset quality:")
    print(f"     python clean_dataset.py --dataset_dir {output_dir.name}")
    print(f"  2. Train model:")
    print(f"     python ../../training/train_synthetic.py \\")
    print(f"       --dataset_dir {output_dir.name} \\")
    print(f"       --output_model_dir ../../models/trocr_mixed \\")
    print(f"       --max_epochs 5")
    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--numbers_dir", required=True,
                       help="Path to numbers dataset directory")
    parser.add_argument("--synthetic_dir", required=True,
                       help="Path to synthetic menu dataset directory")
    parser.add_argument("--output_dir", required=True,
                       help="Path to output mixed dataset directory")
    parser.add_argument("--numbers_ratio", type=float, default=0.7,
                       help="Ratio of numbers samples (0.0 to 1.0, default: 0.7)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    if not (0.0 < args.numbers_ratio < 1.0):
        print("Error: --numbers_ratio must be between 0.0 and 1.0")
        return
    
    try:
        mix_datasets(
            args.numbers_dir,
            args.synthetic_dir,
            args.output_dir,
            args.numbers_ratio,
            args.seed
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
