"""
Stage 4 (recommended): compare baseline vs fine-tuned model on a held-out
set, using the same metric training used (CER) plus WER, so you have a
concrete before/after number before swapping the model into your pipeline.

Usage:
    python evaluate.py --out_dir dataset/ --model microsoft/trocr-base-handwritten
    python evaluate.py --out_dir dataset/ --model my_trocr_menu_v1
"""
import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import evaluate as hf_evaluate

SPECIAL_LABELS = {"<JUNK>", "<HEADER>", ""}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", required=True, help="HF checkpoint name or local fine-tuned model dir")
    parser.add_argument("--split", default="val", help="Which manifest 'split' rows to score; scores everything if the split column is unused")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    with open(out_dir / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows = [r for r in rows if r["text"] not in SPECIAL_LABELS]
    if any(r["split"] for r in rows):
        rows = [r for r in rows if r["split"] == args.split]

    if not rows:
        raise SystemExit("No rows to evaluate on.")

    print(f"Evaluating on {len(rows)} examples with model '{args.model}'")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device)
    model.eval()

    cer_metric = hf_evaluate.load("cer")
    wer_metric = hf_evaluate.load("wer")

    predictions, references = [], []
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

    cer = cer_metric.compute(predictions=predictions, references=references)
    wer = wer_metric.compute(predictions=predictions, references=references)

    print(f"\nCER: {cer:.3%}")
    print(f"WER: {wer:.3%}")

    print("\nSample predictions:")
    for p, r in list(zip(predictions, references))[:10]:
        flag = "" if p == r else "  <-- mismatch"
        print(f"  pred={p!r:30s} true={r!r}{flag}")


if __name__ == "__main__":
    main()
