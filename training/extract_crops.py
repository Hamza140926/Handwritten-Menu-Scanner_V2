"""
Stage 1 of building your own training set: run your EXISTING preprocessing
+ detection stages over a folder of raw menu photos and dump every detected
text region as its own small image, plus a manifest.csv you'll fill in with
the correct transcription in stage 2 (label_tool.py).

This deliberately reuses preprocessing.py and detection.py from your
pipeline's src/ folder, so the crops you label look EXACTLY like what
recognition.py sees at inference time (same deskew, same denoise, same
detector, same perspective-warped crop). Labeling on mismatched
preprocessing is a common, hard-to-notice way fine-tuning underperforms.

Usage:
    python extract_crops.py --images_dir raw_menus/ --out_dir dataset/ --src_dir ../src

Produces:
    dataset/crops/<image_stem>_<region_idx>.png   (one file per detected region)
    dataset/manifest.csv   columns: crop_path,text,split
        "text"  starts empty  - fill in with label_tool.py
        "split" starts empty  - assign "train"/"val"/"test" yourself, or
                                 leave blank and let train_trocr.py split
                                 randomly

Safe to re-run: if you add more raw photos later, re-run with the same
--out_dir and it will only append new crops, not touch existing labels.
"""
import argparse
import csv
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images_dir", required=True, help="Folder of raw menu photos")
    parser.add_argument("--out_dir", required=True, help="Where to write crops/ and manifest.csv")
    parser.add_argument("--src_dir", default="../src", help="Path to your pipeline's src/ folder")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.src_dir).resolve()))
    import cv2
    from preprocessing import preprocess_image
    from detection import detect_text_regions

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        print(f"No images found in {images_dir}")
        return

    manifest_path = out_dir / "manifest.csv"
    existing_rows = []
    if manifest_path.exists():
        with open(manifest_path, newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))
        print(f"Found existing manifest with {len(existing_rows)} rows - will append new crops only.")

    already_have = {row["crop_path"] for row in existing_rows}
    new_rows = []

    for img_path in image_paths:
        print(f"Processing {img_path.name} ...")
        try:
            prep = preprocess_image(str(img_path))
            regions = detect_text_regions(prep["image"])
        except Exception as e:
            print(f"  skipped ({e})")
            continue

        added = 0
        for i, region in enumerate(regions):
            crop_name = f"{img_path.stem}_{i:03d}.png"
            rel_path = str(Path("crops") / crop_name)
            if rel_path in already_have:
                continue
            cv2.imwrite(str(crops_dir / crop_name), region["crop"])
            new_rows.append({"crop_path": rel_path, "text": "", "split": ""})
            added += 1

        print(f"  {len(regions)} regions detected, {added} new")

    all_rows = existing_rows + new_rows
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_path", "text", "split"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(new_rows)} new crops ({len(all_rows)} total) to {manifest_path}")
    print(f"Next: python label_tool.py --out_dir {out_dir}")


if __name__ == "__main__":
    main()
