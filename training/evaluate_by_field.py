"""
Breaks down CER/WER by field type (name / price / header) instead of one
aggregate number, plus shows the worst individual mismatches per type -
so you can see the actual character confusions (e.g. "3"->"8"), not just
a score.

Works with two kinds of manifest.csv:
    1. The synthetic dataset's manifest (from generate_synthetic_dataset.py)
       - already has a "field_type" column, used directly.
    2. A real-photo manifest (from extract_crops.py + label_tool.py) -
       these only have crop_path/text/split, no field_type. In that case
       field type is INFERRED from the ground-truth text using a simple
       heuristic (mostly digits/currency punctuation -> "price", else
       "name"). This is clearly labeled as inferred, not ground truth -
       good enough for a diagnostic breakdown, not for anything you'd
       report as an official metric.

Usage:
    python evaluate_by_field.py --out_dir real_test/ --model ../models/trocr_menu_v1_epoch2
    python evaluate_by_field.py --out_dir ../synthetic/dataset_synth --model ../models/trocr_menu_v1_epoch2 --split test
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import evaluate as hf_evaluate

SPECIAL_LABELS = {"<JUNK>", "<HEADER>", ""}

# mostly digits/decimal separators/currency marks, little or no letters
PRICE_LIKE = re.compile(r"^[\d\s.,€$]+(?:\s*(?:DT|TND|dt))?$", re.IGNORECASE)


def infer_field_type(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "unknown"
    if PRICE_LIKE.match(stripped):
        return "price (inferred)"
    return "name (inferred)"


def load_rows(manifest_path: Path, split: str):
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fieldnames = rows[0].keys() if rows else []
    has_field_type = "field_type" in fieldnames

    rows = [r for r in rows if r["text"] not in SPECIAL_LABELS]
    if any(r.get("split") for r in rows):
        rows = [r for r in rows if r["split"] == split]

    for r in rows:
        r["_field_type"] = r["field_type"] if has_field_type and r.get("field_type") else infer_field_type(r["text"])

    return rows, has_field_type


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--show_worst", type=int, default=8, help="How many worst mismatches to print per field type")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rows, has_field_type = load_rows(out_dir / "manifest.csv", args.split)
    if not rows:
        raise SystemExit(f"No usable rows for split={args.split!r}")

    print(f"{len(rows)} examples ({'using manifest field_type column' if has_field_type else 'field_type INFERRED from text - heuristic, not ground truth'})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device)
    model.eval()

    cer_metric = hf_evaluate.load("cer")
    wer_metric = hf_evaluate.load("wer")

    predictions, references, field_types = [], [], []
    batch_size = 8
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        images = [Image.open(out_dir / r["crop_path"]).convert("RGB") for r in batch]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated = model.generate(pixel_values, max_new_tokens=32)
        texts = processor.batch_decode(generated, skip_special_tokens=True)
        predictions.extend(t.strip() for t in texts)
        references.extend(r["text"] for r in batch)
        field_types.extend(r["_field_type"] for r in batch)

    print(f"\n{'=' * 60}")
    print(f"OVERALL: CER={cer_metric.compute(predictions=predictions, references=references):.2%}  "
          f"WER={wer_metric.compute(predictions=predictions, references=references):.2%}")
    print(f"{'=' * 60}\n")

    by_type = defaultdict(lambda: {"preds": [], "refs": []})
    for p, r, ft in zip(predictions, references, field_types):
        by_type[ft]["preds"].append(p)
        by_type[ft]["refs"].append(r)

    for ft in sorted(by_type, key=lambda k: -len(by_type[k]["refs"])):
        preds = by_type[ft]["preds"]
        refs = by_type[ft]["refs"]
        cer = cer_metric.compute(predictions=preds, references=refs)
        wer = wer_metric.compute(predictions=preds, references=refs)
        print(f"[{ft}] n={len(refs)}  CER={cer:.2%}  WER={wer:.2%}")

        mismatches = [(p, r) for p, r in zip(preds, refs) if p != r]
        mismatches.sort(key=lambda pr: -abs(len(pr[0]) - len(pr[1])))  # roughest mismatches first, cheap proxy
        for p, r in mismatches[:args.show_worst]:
            print(f"    pred={p!r:20s} true={r!r}")
        print()

    print(
        "Read this as: if 'price' CER is meaningfully higher than 'name' CER, "
        "that's a structural signal (digits have no language-model prior to "
        "fall back on, unlike words), not just noise. See the printed mismatches "
        "for the actual character confusions (e.g. which digits get swapped)."
    )


if __name__ == "__main__":
    main()