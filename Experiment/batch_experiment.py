"""
batch_experiment.py

Runs the tilted horizontal line pairing algorithm on all images in the input/
folder and outputs debug visualizations to debug_output/.

Usage:
    python batch_experiment.py
    python batch_experiment.py --force-classical  # skip PaddleOCR
"""

import sys
import os
import argparse
from pathlib import Path
import cv2

# Import the experiment logic
from pairing_experiment import (
    get_boxes,
    assign_reading_order,
    classify_categories,
    run_pairing,
    assign_groups,
    detect_vertical_dividers,
    print_report,
    draw_debug,
)


def process_image(image_path: Path, output_dir: Path, force_classical: bool) -> dict:
    """
    Process a single image through the pairing experiment.
    
    Returns dict with:
        - success: bool
        - error: str (if failed)
        - boxes_found: int
        - pairs_found: int
        - orphans: int
        - categories: int
    """
    print(f"\n{'='*70}")
    print(f"Processing: {image_path.name}")
    print('='*70)
    
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return {"success": False, "error": "Could not read image"}
        
        # Run the pairing algorithm
        boxes, source = get_boxes(image, force_classical)
        print(f"[info] detector: {source}, boxes: {len(boxes)}")
        
        if not boxes:
            return {"success": False, "error": "0 boxes detected"}
        
        boxes = assign_reading_order(boxes)
        median_item_height = classify_categories(boxes)
        run_pairing(boxes, median_item_height)
        categories = assign_groups(boxes)
        verticals = detect_vertical_dividers(image)
        
        # Print report for this image
        print_report(boxes, categories)
        
        # Save debug visualization
        output_path = output_dir / f"{image_path.stem}_pairing_debug.png"
        debug_img = draw_debug(image, boxes, verticals)
        cv2.imwrite(str(output_path), debug_img)
        print(f"\n[info] debug image saved to {output_path}")
        
        # Calculate stats
        pairs = sum(1 for b in boxes if b.paired_with is not None and b.kind == "item") // 2
        orphans = sum(1 for b in boxes if b.paired_with is None and b.kind == "item")
        
        return {
            "success": True,
            "boxes_found": len(boxes),
            "pairs_found": pairs,
            "orphans": orphans,
            "categories": len(categories),
            "detector": source,
        }
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to process {image_path.name}: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Batch process menu images")
    parser.add_argument("--force-classical", action="store_true",
                       help="Skip PaddleOCR and use classical CV detector")
    parser.add_argument("--input-dir", default="input",
                       help="Input directory (default: input/)")
    parser.add_argument("--output-dir", default="debug_output",
                       help="Output directory (default: debug_output/)")
    args = parser.parse_args()
    
    # Setup directories
    script_dir = Path(__file__).parent
    input_dir = script_dir / args.input_dir
    output_dir = script_dir / args.output_dir
    
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        sys.exit(1)
    
    output_dir.mkdir(exist_ok=True)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [
        f for f in input_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"[ERROR] No images found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images to process")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    # Process all images
    results = []
    for img_path in sorted(image_files):
        result = process_image(img_path, output_dir, args.force_classical)
        result["filename"] = img_path.name
        results.append(result)
    
    # Summary report
    print(f"\n{'='*70}")
    print("BATCH PROCESSING SUMMARY")
    print('='*70)
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"\nTotal images: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\n{'Filename':<20} {'Boxes':<8} {'Pairs':<8} {'Orphans':<10} {'Categories':<12}")
        print('-' * 70)
        for r in successful:
            print(f"{r['filename']:<20} {r['boxes_found']:<8} "
                  f"{r['pairs_found']:<8} {r['orphans']:<10} {r['categories']:<12}")
        
        # Aggregate stats
        total_boxes = sum(r["boxes_found"] for r in successful)
        total_pairs = sum(r["pairs_found"] for r in successful)
        total_orphans = sum(r["orphans"] for r in successful)
        avg_orphan_rate = (total_orphans / total_boxes * 100) if total_boxes > 0 else 0
        
        print('-' * 70)
        print(f"{'TOTALS':<20} {total_boxes:<8} {total_pairs:<8} {total_orphans:<10}")
        print(f"\nOrphan Rate: {avg_orphan_rate:.1f}%")
    
    if failed:
        print(f"\n{'='*70}")
        print("FAILED IMAGES")
        print('='*70)
        for r in failed:
            print(f"{r['filename']}: {r['error']}")
    
    print(f"\nAll debug images saved to: {output_dir}")


if __name__ == "__main__":
    main()
