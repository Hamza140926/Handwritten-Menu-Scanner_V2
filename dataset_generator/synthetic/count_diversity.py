"""
Measures actual text diversity in a manifest.csv - the real number that
matters for whether a train/test split can be meaningful at all, not
just total row count.

Usage:
    python count_diversity.py --dataset_dir dataset_numbers
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

SPECIAL_LABELS = {"<JUNK>", "<HEADER>", ""}


def main():
    print("count_diversity.py starting...", flush=True)  # if this doesn't print, it's not a script bug - python itself isn't running it

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_dir", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    manifest_path = dataset_dir / "manifest.csv"
    print(f"Looking for manifest at: {manifest_path.resolve()}", flush=True)

    if not manifest_path.exists():
        print(f"ERROR: manifest.csv not found at {manifest_path.resolve()}")
        return

    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    usable = [r for r in rows if r["text"] not in SPECIAL_LABELS]
    print(f"Total rows: {len(rows)}  (usable, non-junk/header: {len(usable)})\n")

    all_texts = [r["text"] for r in usable]
    counts = Counter(all_texts)
    distinct = len(counts)

    print(f"DISTINCT text strings: {distinct}")
    print(f"Total usable rows:     {len(usable)}")
    print(f"Average repeats per distinct string: {len(usable) / max(distinct, 1):.1f}")
    print()

    print("Most repeated strings (top 10):")
    for text, count in counts.most_common(10):
        print(f"  {count:4d}x  {text!r}")
    print()

    # Per-split distinct counts, if split column exists
    if any(r.get("split") for r in usable):
        print("Per-split distinct text counts:")
        by_split = {}
        for r in usable:
            by_split.setdefault(r["split"], []).append(r["text"])
        for split, texts in by_split.items():
            print(f"  {split}: {len(texts)} rows, {len(set(texts))} distinct texts")
        print()

    # The actual verdict
    print("=" * 60)
    if distinct < 100:
        print(
            f"LOW DIVERSITY ({distinct} distinct strings): with this few unique "
            f"values, any train/test split will almost inevitably share most or "
            f"all text between splits - there just isn't enough content to hold "
            f"anything out. This alone can explain near-zero CER on 'test' - the "
            f"model isn't generalizing, it's seen these exact values before under "
            f"a different split label."
        )
    elif distinct < 300:
        print(
            f"MODERATE DIVERSITY ({distinct} distinct strings): a real split is "
            f"possible but tight - worth checking the leakage checker's Check 3 "
            f"result (% of test text also in train) stays low after re-splitting."
        )
    else:
        print(f"OK DIVERSITY ({distinct} distinct strings) - a meaningful disjoint split should be achievable.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\nSCRIPT CRASHED: {e}")
        traceback.print_exc()