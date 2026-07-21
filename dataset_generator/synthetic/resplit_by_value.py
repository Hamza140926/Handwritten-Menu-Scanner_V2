"""
Fixes the numbers dataset's train/val/test split to be VALUE-DISJOINT:
every row sharing the same exact text goes entirely into one split, never
split across train/val/test. This directly fixes the leakage confirmed by
count_diversity.py (train already covered 250/254 distinct values) and
check_leakage.py (89% of test texts also present in train).

This does NOT touch images or crop_path - only rewrites the 'split'
column in manifest.csv. A .backup of the original is kept automatically
(if one doesn't already exist - this script won't overwrite an existing
.backup from a previous run).

Usage:
    python resplit_by_value.py --dataset_dir dataset_numbers

    Then re-run the mixing + training pipeline:
    python mix_datasets.py --numbers_dir dataset_numbers --synthetic_dir dataset_synth --output_dir dataset_mixed --numbers_ratio 0.7
    python train_synthetic.py --dataset_dir dataset_mixed --base_checkpoint ../models/trocr_menu_v1_epoch2 --output_model_dir ../models/trocr_menu_v1_digits_v3 --batch_size 4 --gradient_accumulation_steps 1 --max_epochs 3 --lr 1e-5 --eval_test
    python check_leakage.py --dataset_dir dataset_mixed   # confirm 0 cross-split text overlap this time
"""
import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

SPECIAL_LABELS = {"<JUNK>", "<HEADER>", ""}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    manifest_path = dataset_dir / "manifest.csv"
    backup_path = dataset_dir / "manifest.csv.backup_before_resplit"

    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys()) if rows else []

    if not backup_path.exists():
        shutil.copy2(manifest_path, backup_path)
        print(f"Backed up original manifest to {backup_path}")
    else:
        print(f"Backup already exists at {backup_path} - not overwriting.")

    # Group rows by exact text (special-label rows keep split="" / stay excluded downstream)
    by_text = defaultdict(list)
    special_rows = []
    for r in rows:
        if r["text"] in SPECIAL_LABELS:
            special_rows.append(r)
        else:
            by_text[r["text"]].append(r)

    distinct_texts = list(by_text.keys())
    print(f"\n{len(rows)} total rows, {len(distinct_texts)} distinct usable text values")

    rng = random.Random(args.seed)
    rng.shuffle(distinct_texts)

    n = len(distinct_texts)
    n_test = max(1, int(n * args.test_fraction))
    n_val = max(1, int(n * args.val_fraction))
    test_texts = set(distinct_texts[:n_test])
    val_texts = set(distinct_texts[n_test:n_test + n_val])
    train_texts = set(distinct_texts[n_test + n_val:])

    print(f"\nValue-disjoint split (by DISTINCT TEXT, not by row):")
    print(f"  train: {len(train_texts)} distinct values")
    print(f"  val:   {len(val_texts)} distinct values")
    print(f"  test:  {len(test_texts)} distinct values")

    # Sanity check: these sets must be pairwise disjoint by construction
    assert not (train_texts & val_texts), "BUG: train/val overlap"
    assert not (train_texts & test_texts), "BUG: train/test overlap"
    assert not (val_texts & test_texts), "BUG: val/test overlap"

    new_rows = []
    row_counts = {"train": 0, "val": 0, "test": 0}
    for text, text_rows in by_text.items():
        split = "train" if text in train_texts else ("val" if text in val_texts else "test")
        for r in text_rows:
            r = dict(r)
            r["split"] = split
            new_rows.append(r)
            row_counts[split] += 1

    for r in special_rows:
        r = dict(r)
        # keep special rows in whatever split they were already in - they're
        # excluded from training/eval anyway (see SPECIAL_LABELS), so their
        # split assignment doesn't affect leakage
        new_rows.append(r)

    print(f"\nRow counts after re-split:")
    print(f"  train: {row_counts['train']} rows")
    print(f"  val:   {row_counts['val']} rows")
    print(f"  test:  {row_counts['test']} rows")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)

    print(f"\nWrote corrected manifest to {manifest_path}")
    print("\nVerification: no distinct text value should now appear in more than one split.")
    check_by_text = defaultdict(set)
    for r in new_rows:
        if r["text"] not in SPECIAL_LABELS:
            check_by_text[r["text"]].add(r["split"])
    leaked = {t: s for t, s in check_by_text.items() if len(s) > 1}
    if leaked:
        print(f"  FAILED: {len(leaked)} values still span multiple splits: {list(leaked.items())[:5]}")
    else:
        print("  PASSED: every distinct value belongs to exactly one split.")

    print(
        "\nNext: re-run mix_datasets.py with this corrected manifest, then retrain "
        "from ../models/trocr_menu_v1_epoch2 (not from the leaky digits_v2 checkpoint - "
        "that one already memorized the old leaky split during training)."
    )


if __name__ == "__main__":
    main()