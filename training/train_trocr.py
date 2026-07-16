"""
Stage 3: fine-tune TrOCR on your labeled dataset.

Starts from the same pretrained checkpoint your pipeline already uses
(microsoft/trocr-base-handwritten, per src/config.py's RecognitionConfig),
so you're adapting a strong handwriting baseline to your menus' specific
handwriting and vocabulary rather than training from scratch - this
matters a lot with a small dataset.

Usage:
    python train_trocr.py --out_dir dataset/ --output_model_dir my_trocr_menu_v1

What it does:
    - Loads dataset/manifest.csv, drops rows marked <JUNK>, <HEADER>, or
      still unlabeled
    - Splits into train/val (80/20 by default, or uses the "split" column
      if you filled it in yourself)
    - Fine-tunes with Hugging Face's Seq2SeqTrainer, evaluating Character
      Error Rate (CER) each epoch, keeping the best checkpoint, with early
      stopping
    - Saves the final model in standard transformers format, so it's a
      drop-in replacement in your pipeline (see README.md)

A note on dataset size:
    Fine-tuning (not training from scratch) can show real gains with a
    few hundred labeled crops, more with 1000+. Under ~100 examples,
    expect high variance - the val CER won't be very reliable and it's
    easy to overfit. The script will warn you if your labeled set is
    small; treat early results as directional, not final.
"""
import argparse
import csv
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

SPECIAL_LABELS = {"<JUNK>", "<HEADER>", ""}


class MenuCropDataset(Dataset):
    def __init__(self, rows, images_root: Path, processor, max_target_length=32):
        self.rows = rows
        self.images_root = images_root
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(self.images_root / row["crop_path"]).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            row["text"],
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids
        # Ignore pad tokens in the loss
        labels = [l if l != self.processor.tokenizer.pad_token_id else -100 for l in labels]
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


def load_rows(manifest_path: Path):
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["text"] not in SPECIAL_LABELS]


def split_rows(rows, val_fraction=0.2, seed=42):
    if rows and all(r["split"] in {"train", "val"} for r in rows):
        return [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "val"]

    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


def make_compute_metrics(processor):
    import evaluate
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"cer": cer_metric.compute(predictions=pred_str, references=label_str)}

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", required=True, help="Same dataset dir used in extract_crops.py / label_tool.py")
    parser.add_argument("--base_checkpoint", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--output_model_dir", required=True)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--early_stopping_patience", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rows = load_rows(out_dir / "manifest.csv")
    print(f"{len(rows)} usable labeled examples (junk/header/unlabeled excluded).")
    if len(rows) < 30:
        print(
            "WARNING: fewer than 30 labeled examples. Fine-tuning will be "
            "very noisy at this size - strongly consider labeling more "
            "crops before trusting these results."
        )

    train_rows, val_rows = split_rows(rows, args.val_fraction)
    print(f"Train: {len(train_rows)}  Val: {len(val_rows)}")
    if not val_rows:
        raise SystemExit("No validation rows - label more data before training.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    if device == "cpu":
        print("No GPU detected by torch - this will be very slow. Check your CUDA/torch install.")

    processor = TrOCRProcessor.from_pretrained(args.base_checkpoint)
    model = VisionEncoderDecoderModel.from_pretrained(args.base_checkpoint).to(device)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 32
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    train_dataset = MenuCropDataset(train_rows, out_dir, processor)
    val_dataset = MenuCropDataset(val_rows, out_dir, processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(Path(args.output_model_dir) / "checkpoints"),
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.max_epochs,
        learning_rate=args.lr,
        fp16=(device == "cuda"),
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        save_total_limit=2,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=make_compute_metrics(processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()

    final_dir = Path(args.output_model_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\nSaved fine-tuned model to {final_dir}")
    print(
        "To use it: set RecognitionConfig.model_checkpoint to "
        f"'{final_dir}' in src/config.py (see README.md)."
    )


if __name__ == "__main__":
    main()
