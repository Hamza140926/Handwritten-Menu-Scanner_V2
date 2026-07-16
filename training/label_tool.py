"""
Stage 2: label the crops extracted by extract_crops.py.

Shows each unlabeled crop in an image window; you type the correct
transcription in the terminal and hit Enter. Progress is saved to disk
after every single label, so it's safe to stop (Ctrl+C) and resume anytime
- just re-run the same command.

Controls (typed in the terminal, not the image window):
    <the text you see> + Enter   -> save as this crop's transcription
    (empty) + Enter               -> skip for now, ask again next run
    "header" + Enter              -> mark as a category header / non-item
                                      text (e.g. "Coffee"), saved as
                                      "<HEADER>" - train_trocr.py excludes
                                      these from training by default
    "junk" + Enter                 -> mark as unusable (bad crop, blank,
                                      illegible even to you), saved as
                                      "<JUNK>" and excluded from training
    "quit" + Enter                 -> stop, progress already saved

Usage:
    python label_tool.py --out_dir dataset/

Requires a display (this opens an OpenCV image window) - run it on your
own machine, not over a headless SSH session without X forwarding.
"""
import argparse
import csv
from pathlib import Path
import cv2

FIELDS = ["crop_path", "text", "split"]


def load_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_manifest(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "manifest.csv"
    rows = load_manifest(manifest_path)

    unlabeled = [r for r in rows if not r["text"]]
    print(f"{len(unlabeled)} unlabeled crops out of {len(rows)} total.")
    if not unlabeled:
        print("Nothing left to label.")
        return

    labeled_count = len(rows) - len(unlabeled)

    try:
        for row in rows:
            if row["text"]:
                continue

            crop_path = out_dir / row["crop_path"]
            image = cv2.imread(str(crop_path))
            if image is None:
                print(f"Could not load {crop_path}, marking as junk.")
                row["text"] = "<JUNK>"
                save_manifest(manifest_path, rows)
                continue

            # Upscale small crops so handwriting is actually readable
            h, w = image.shape[:2]
            scale = max(1, 400 // max(h, 1))
            display = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("label_tool - type the transcription in the terminal", display)
            cv2.waitKey(1)

            text = input(f"[{labeled_count}/{len(rows)}] {row['crop_path']} > ").strip()

            if text.lower() == "quit":
                print("Stopping. Progress saved.")
                break
            elif text.lower() == "header":
                row["text"] = "<HEADER>"
            elif text.lower() == "junk":
                row["text"] = "<JUNK>"
            elif text == "":
                continue
            else:
                row["text"] = text

            labeled_count += 1
            save_manifest(manifest_path, rows)
    finally:
        cv2.destroyAllWindows()

    remaining = sum(1 for r in rows if not r["text"])
    print(f"\n{labeled_count} labeled so far. {remaining} crops still unlabeled.")


if __name__ == "__main__":
    main()
