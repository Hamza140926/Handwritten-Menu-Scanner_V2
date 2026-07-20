"""
Manual dataset cleaning tool.

Displays each image with its label and allows you to mark samples as valid or invalid.

Controls:
- ENTER: Mark as valid (keep the sample)
- X: Mark as invalid (delete the sample)
- Q: Quit and save progress
- S: Skip to next without decision
- B: Go back to previous sample

Usage:
    python clean_dataset.py --dataset_dir dataset_numbers
    
The script will:
1. Load manifest.csv
2. Display each image with its label
3. Let you mark valid/invalid
4. Delete invalid images and update manifest.csv
"""
import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np


class DatasetCleaner:
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.manifest_path = self.dataset_dir / "manifest.csv"
        
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        
        # Load manifest
        self.samples = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.samples = list(reader)
        
        print(f"Loaded {len(self.samples)} samples from manifest")
        
        # Track decisions
        self.valid = []
        self.invalid = []
        self.current_idx = 0
        
    def display_sample(self, idx):
        """Display a sample and return user decision."""
        if idx < 0 or idx >= len(self.samples):
            return None
        
        sample = self.samples[idx]
        img_path = self.dataset_dir / sample["crop_path"]
        
        if not img_path.exists():
            print(f"\n[ERROR] Image not found: {img_path}")
            return "skip"
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"\n[ERROR] Could not load image: {img_path}")
            return "skip"
        
        # Create display with info
        h, w = img.shape[:2]
        
        # Add padding and info text
        padding = 100
        display = np.ones((h + padding, max(w, 600), 3), dtype=np.uint8) * 255
        display[padding:padding+h, 0:w] = img
        
        # Add text info
        text_label = f"Label: '{sample['text']}'"
        text_split = f"Split: {sample['split']}"
        text_style = f"Style: {sample['style_id']} | Bias: {sample['bias']}"
        text_progress = f"Progress: {idx + 1}/{len(self.samples)}"
        text_stats = f"Valid: {len(self.valid)} | Invalid: {len(self.invalid)}"
        
        cv2.putText(display, text_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 0), 2)
        cv2.putText(display, text_split, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (100, 100, 100), 1)
        cv2.putText(display, text_style, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (100, 100, 100), 1)
        cv2.putText(display, text_progress, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 255), 1)
        cv2.putText(display, text_stats, (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 150, 0), 1)
        
        # Instructions
        instructions = "[ENTER] Valid | [X] Invalid | [S] Skip | [B] Back | [Q] Quit"
        cv2.putText(display, instructions, (10, h + padding - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Show image
        cv2.imshow("Dataset Cleaner", display)
        
        # Wait for key press
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            # Enter or Space - Valid
            if key == 13 or key == 32:  # Enter or Space
                return "valid"
            
            # X or Delete - Invalid
            elif key == ord('x') or key == ord('X') or key == 127:
                return "invalid"
            
            # S - Skip
            elif key == ord('s') or key == ord('S'):
                return "skip"
            
            # B - Back
            elif key == ord('b') or key == ord('B'):
                return "back"
            
            # Q or Esc - Quit
            elif key == ord('q') or key == ord('Q') or key == 27:
                return "quit"
    
    def run(self):
        """Run the cleaning process."""
        print("\n" + "=" * 70)
        print("DATASET CLEANER")
        print("=" * 70)
        print("\nControls:")
        print("  [ENTER/SPACE] - Mark as VALID (keep)")
        print("  [X]           - Mark as INVALID (delete)")
        print("  [S]           - Skip (no decision)")
        print("  [B]           - Go back to previous")
        print("  [Q/ESC]       - Quit and save")
        print("\nStarting review...\n")
        
        cv2.namedWindow("Dataset Cleaner", cv2.WINDOW_NORMAL)
        
        while self.current_idx < len(self.samples):
            decision = self.display_sample(self.current_idx)
            
            if decision == "valid":
                self.valid.append(self.current_idx)
                print(f"[{self.current_idx + 1}/{len(self.samples)}] ✓ Valid")
                self.current_idx += 1
            
            elif decision == "invalid":
                self.invalid.append(self.current_idx)
                print(f"[{self.current_idx + 1}/{len(self.samples)}] ✗ Invalid")
                self.current_idx += 1
            
            elif decision == "skip":
                print(f"[{self.current_idx + 1}/{len(self.samples)}] → Skipped")
                self.current_idx += 1
            
            elif decision == "back":
                if self.current_idx > 0:
                    self.current_idx -= 1
                    # Remove previous decision if it exists
                    if self.current_idx in self.valid:
                        self.valid.remove(self.current_idx)
                    if self.current_idx in self.invalid:
                        self.invalid.remove(self.current_idx)
                    print(f"[{self.current_idx + 1}/{len(self.samples)}] ← Back")
                else:
                    print("Already at first sample")
            
            elif decision == "quit":
                print("\nQuitting...")
                break
        
        cv2.destroyAllWindows()
        
        # Process results
        self.save_results()
    
    def save_results(self):
        """Delete invalid samples and update manifest."""
        print("\n" + "=" * 70)
        print("CLEANING RESULTS")
        print("=" * 70)
        print(f"\nTotal samples reviewed: {len(self.valid) + len(self.invalid)}")
        print(f"  Valid: {len(self.valid)}")
        print(f"  Invalid: {len(self.invalid)}")
        print(f"  Skipped: {len(self.samples) - len(self.valid) - len(self.invalid)}")
        
        if not self.invalid:
            print("\nNo invalid samples to delete. Exiting.")
            return
        
        # Confirm deletion
        print(f"\nAbout to DELETE {len(self.invalid)} invalid samples.")
        response = input("Continue? [y/N]: ").strip().lower()
        
        if response != 'y':
            print("Deletion cancelled. No changes made.")
            return
        
        # Delete invalid images
        print("\nDeleting invalid samples...")
        deleted_count = 0
        for idx in self.invalid:
            sample = self.samples[idx]
            img_path = self.dataset_dir / sample["crop_path"]
            
            if img_path.exists():
                try:
                    os.remove(img_path)
                    deleted_count += 1
                    print(f"  Deleted: {img_path.name}")
                except Exception as e:
                    print(f"  [ERROR] Could not delete {img_path.name}: {e}")
        
        # Update manifest - keep only samples not in invalid set
        invalid_set = set(self.invalid)
        cleaned_samples = [
            sample for i, sample in enumerate(self.samples) 
            if i not in invalid_set
        ]
        
        # Backup original manifest
        backup_path = self.manifest_path.with_suffix(".csv.backup")
        print(f"\nBacking up original manifest to: {backup_path}")
        os.replace(self.manifest_path, backup_path)
        
        # Write cleaned manifest
        print(f"Writing cleaned manifest: {self.manifest_path}")
        with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
            if cleaned_samples:
                writer = csv.DictWriter(f, fieldnames=cleaned_samples[0].keys())
                writer.writeheader()
                writer.writerows(cleaned_samples)
        
        # Final stats
        train_count = sum(1 for s in cleaned_samples if s["split"] == "train")
        val_count = sum(1 for s in cleaned_samples if s["split"] == "val")
        test_count = sum(1 for s in cleaned_samples if s["split"] == "test")
        
        print("\n" + "=" * 70)
        print("CLEANING COMPLETE")
        print("=" * 70)
        print(f"\nOriginal samples: {len(self.samples)}")
        print(f"Deleted samples:  {deleted_count}")
        print(f"Remaining samples: {len(cleaned_samples)}")
        print(f"\nSplit distribution:")
        print(f"  Train: {train_count}")
        print(f"  Val:   {val_count}")
        print(f"  Test:  {test_count}")
        print(f"\nOriginal manifest backed up to: {backup_path}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset_dir", required=True, 
                       help="Path to dataset directory (containing manifest.csv)")
    args = parser.parse_args()
    
    try:
        cleaner = DatasetCleaner(args.dataset_dir)
        cleaner.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress not saved.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
