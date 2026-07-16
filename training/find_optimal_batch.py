"""
Find optimal batch size for your GPU by testing different sizes.
This will help maximize GPU utilization on RTX 3050 4GB.

Usage:
    python find_optimal_batch.py --dataset_dir ../dataset_generator/synthetic/dataset_synth
"""
import argparse
import csv
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import time


class QuickDataset(Dataset):
    def __init__(self, rows, dataset_root, processor, max_samples=50):
        self.rows = rows[:max_samples]
        self.dataset_root = dataset_root
        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_path = self.dataset_root / row["crop_path"]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(row["text"], padding="max_length", max_length=32, truncation=True).input_ids
        labels = [l if l != self.processor.tokenizer.pad_token_id else -100 for l in labels]
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


def test_batch_size(model, dataset, batch_size, device):
    """Test if this batch size fits in memory and measure speed."""
    try:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        model.train()
        
        # Warm up
        batch = next(iter(dataloader))
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        torch.cuda.synchronize()
        start = time.time()
        
        # Test 5 batches
        for i, batch in enumerate(dataloader):
            if i >= 5:
                break
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            model.zero_grad()
            torch.cuda.empty_cache()
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        samples_per_sec = (5 * batch_size) / elapsed
        mem_allocated = torch.cuda.max_memory_allocated(0) / 1024**2
        mem_reserved = torch.cuda.max_memory_reserved(0) / 1024**2
        
        return True, samples_per_sec, mem_allocated, mem_reserved
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            return False, 0, 0, 0
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--base_checkpoint", default="microsoft/trocr-base-handwritten")
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: No GPU detected. This script requires CUDA GPU.")
        return
    
    device = torch.device("cuda")
    print("=" * 70)
    print("GPU BATCH SIZE OPTIMIZER")
    print("=" * 70)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load model
    print(f"\nLoading model: {args.base_checkpoint}")
    processor = TrOCRProcessor.from_pretrained(args.base_checkpoint)
    model = VisionEncoderDecoderModel.from_pretrained(args.base_checkpoint).to(device)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    
    # Load small dataset sample
    dataset_dir = Path(args.dataset_dir)
    manifest_path = dataset_dir / "manifest.csv"
    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "train"]
    
    dataset = QuickDataset(rows, dataset_dir, processor, max_samples=50)
    print(f"Testing with {len(dataset)} samples")
    
    # Test different batch sizes
    print("\n" + "=" * 70)
    print("TESTING BATCH SIZES")
    print("=" * 70)
    
    batch_sizes = [2, 4, 8, 12, 16, 20, 24, 28, 32]
    results = []
    
    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        print(f"\nTesting batch_size={bs}...", end=" ")
        success, speed, mem_alloc, mem_reserve = test_batch_size(model, dataset, bs, device)
        
        if success:
            print(f"✓ OK")
            print(f"  Speed: {speed:.1f} samples/sec")
            print(f"  Memory allocated: {mem_alloc:.0f} MB")
            print(f"  Memory reserved: {mem_reserve:.0f} MB")
            results.append({
                "batch_size": bs,
                "speed": speed,
                "mem_alloc": mem_alloc,
                "mem_reserve": mem_reserve
            })
        else:
            print(f"✗ OUT OF MEMORY")
            break
    
    if not results:
        print("\n✗ No batch size fits in memory!")
        return
    
    # Find optimal
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    # Sort by speed
    results.sort(key=lambda x: x["speed"], reverse=True)
    best = results[0]
    
    print(f"\n🏆 OPTIMAL BATCH SIZE: {best['batch_size']}")
    print(f"   Speed: {best['speed']:.1f} samples/sec")
    print(f"   VRAM usage: {best['mem_reserve']:.0f} MB / ~4096 MB ({best['mem_reserve']/4096*100:.1f}%)")
    
    # Suggest gradient accumulation
    if best['batch_size'] < 16:
        grad_accum = max(1, 16 // best['batch_size'])
        effective = best['batch_size'] * grad_accum
        print(f"\n💡 For effective batch size of {effective}:")
        print(f"   --batch_size {best['batch_size']} --gradient_accumulation_steps {grad_accum}")
    
    print(f"\n📋 TRAINING COMMAND:")
    print(f"""
python train_synthetic.py \\
    --dataset_dir {args.dataset_dir} \\
    --output_model_dir ../models/trocr_menu_v1 \\
    --batch_size {best['batch_size']} \\
    --gradient_accumulation_steps 2 \\
    --max_epochs 20 \\
    --eval_test
""")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
