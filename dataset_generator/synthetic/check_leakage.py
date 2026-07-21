"""
Checks a combined (mixed) manifest.csv for the two most likely causes of
a suspicious CER: 0.0000 result: exact text leakage and exact/near-exact
image duplication between train and test (or train and val).

Run this BEFORE trusting any eval number from a mixed dataset - it's the
fastest way to confirm or rule out leakage as the explanation for an
implausibly perfect score.

Usage:
    python check_leakage.py --dataset_dir dataset_mixed
"""
import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path


def load_manifest(manifest_path: Path):
    with open(manifest_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def file_hash(path: Path, chunk_size=8192) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_dir", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    rows = load_manifest(dataset_dir / "manifest.csv")
    print(f"{len(rows)} total rows\n")

    by_split = defaultdict(list)
    for r in rows:
        by_split[r.get("split", "unknown")].append(r)

    for split, split_rows in by_split.items():
        print(f"  {split}: {len(split_rows)} rows")
    print()

    # --- Check 1: exact text overlap between splits ---
    print("=" * 60)
    print("CHECK 1: exact text overlap between splits")
    print("=" * 60)
    texts_by_split = {split: set(r["text"] for r in split_rows) for split, split_rows in by_split.items()}

    for a in texts_by_split:
        for b in texts_by_split:
            if a >= b:
                continue
            overlap = texts_by_split[a] & texts_by_split[b]
            pct_of_b = len(overlap) / max(len(texts_by_split[b]), 1)
            flag = "  <-- SUSPICIOUSLY HIGH" if pct_of_b > 0.3 else ""
            print(f"  {a} <-> {b}: {len(overlap)} shared unique texts ({pct_of_b:.0%} of {b}){flag}")
    print()

    # --- Check 2: exact duplicate images across splits (the real smoking gun) ---
    print("=" * 60)
    print("CHECK 2: exact duplicate IMAGE FILES across splits (byte-for-byte)")
    print("=" * 60)
    print("Hashing all images... (this may take a minute for large datasets)")

    hash_to_rows = defaultdict(list)
    missing = 0
    for r in rows:
        img_path = dataset_dir / r["crop_path"]
        if not img_path.exists():
            missing += 1
            continue
        h = file_hash(img_path)
        hash_to_rows[h].append(r)

    if missing:
        print(f"  ({missing} rows had missing image files, skipped)")

    cross_split_dupes = []
    for h, dupe_rows in hash_to_rows.items():
        splits_involved = set(r.get("split", "unknown") for r in dupe_rows)
        if len(splits_involved) > 1:
            cross_split_dupes.append((h, dupe_rows, splits_involved))

    print(f"\n{len(cross_split_dupes)} exact-duplicate images found spanning multiple splits")
    if cross_split_dupes:
        print("(This alone is enough to explain an implausibly low/zero CER)")
        print("\nExamples:")
        for h, dupe_rows, splits_involved in cross_split_dupes[:10]:
            paths = [r["crop_path"] for r in dupe_rows]
            texts = set(r["text"] for r in dupe_rows)
            print(f"  hash {h[:12]}... appears in splits {splits_involved}")
            print(f"    paths: {paths}")
            print(f"    text(s): {texts}")
    print()

    # --- Check 3: how many test rows have an exact text match somewhere in train ---
    print("=" * 60)
    print("CHECK 3: test rows with EXACT text also present in train")
    print("=" * 60)
    if "test" in texts_by_split and "train" in texts_by_split:
        test_in_train = texts_by_split["test"] & texts_by_split["train"]
        print(f"  {len(test_in_train)} / {len(texts_by_split['test'])} unique test texts "
              f"also appear in train ({len(test_in_train)/max(len(texts_by_split['test']),1):.0%})")
    else:
        print("  No test/train split found to compare.")
    print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if cross_split_dupes:
        print(
            f"LEAKAGE CONFIRMED: {len(cross_split_dupes)} images are byte-identical across "
            f"splits. This alone explains a CER of 0.0000 - the model is being tested on "
            f"images it directly trained on. Fix the split assignment (most likely in "
            f"whichever script generated the source 'numbers' dataset's split column) "
            f"before trusting any eval number from this dataset."
        )
    else:
        print(
            "No exact byte-identical images found across splits. If CER is still "
            "implausibly low, check: (1) near-duplicate but not byte-identical images "
            "(e.g. same number re-rendered with tiny variations), (2) whether the source "
            "'numbers' dataset has enough unique underlying content for its split to be "
            "meaningful at all - a small pool of distinct digit strings repeated many "
            "times can still make memorization trivial even without literal duplicates."
        )


if __name__ == "__main__":
    main()