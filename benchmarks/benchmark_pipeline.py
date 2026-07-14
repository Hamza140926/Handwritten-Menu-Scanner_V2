"""
Performance benchmarking for the menu scanner pipeline.

Measures execution time and memory usage for each pipeline stage.

Usage:
    python benchmarks/benchmark_pipeline.py path/to/menu.jpg
    
    # Benchmark multiple images
    python benchmarks/benchmark_pipeline.py path/to/menu1.jpg path/to/menu2.jpg
    
    # Save results to JSON
    python benchmarks/benchmark_pipeline.py path/to/menu.jpg --output results.json
"""

import sys
import time
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import preprocess_image
from detection import detect_text_regions
from recognition import recognize_regions
from postprocess import process_recognition_results
from pipeline import assemble_menu, DEFAULT_CURRENCY
from validation import validate_image_input
from logging_config import setup_logging, get_logger

logger = get_logger(__name__)


class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
    
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.duration is None:
            return 0.0
        return self.duration * 1000


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage."""
    memory = {}
    
    if torch.cuda.is_available():
        memory["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
        memory["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
        memory["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
    
    try:
        import psutil
        process = psutil.Process()
        memory["cpu_ram_mb"] = process.memory_info().rss / 1024**2
    except ImportError:
        pass
    
    return memory


def reset_peak_memory():
    """Reset peak memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def benchmark_stage(name: str, func, *args, **kwargs) -> Dict[str, Any]:
    """Benchmark a single pipeline stage."""
    logger.info(f"Benchmarking: {name}")
    
    # Reset memory stats
    reset_peak_memory()
    memory_before = get_memory_usage()
    
    # Time the operation
    with Timer(name) as timer:
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{name} failed: {e}")
            return {
                "name": name,
                "success": False,
                "error": str(e),
                "duration_ms": 0,
            }
    
    memory_after = get_memory_usage()
    
    # Calculate memory delta
    memory_delta = {}
    for key in memory_after:
        if key in memory_before:
            memory_delta[f"{key}_delta"] = memory_after[key] - memory_before[key]
    
    return {
        "name": name,
        "success": True,
        "duration_ms": timer.elapsed_ms(),
        "memory_before": memory_before,
        "memory_after": memory_after,
        "memory_delta": memory_delta,
        "result": result,
    }


def benchmark_full_pipeline(image_path: str, currency: str = DEFAULT_CURRENCY) -> Dict[str, Any]:
    """Benchmark the complete pipeline with detailed timing per stage."""
    
    logger.info(f"Starting benchmark for: {image_path}")
    
    results = {
        "image_path": image_path,
        "currency": currency,
        "stages": {},
        "total_time_ms": 0,
        "success": False,
    }
    
    try:
        # Stage 0: Validation
        stage = benchmark_stage(
            "validation",
            validate_image_input,
            image_path
        )
        results["stages"]["validation"] = {k: v for k, v in stage.items() if k != "result"}
        if not stage["success"]:
            return results
        validated_path = stage["result"]
        
        # Stage 1: Preprocessing
        stage = benchmark_stage(
            "preprocessing",
            preprocess_image,
            str(validated_path)
        )
        results["stages"]["preprocessing"] = {k: v for k, v in stage.items() if k != "result"}
        if not stage["success"]:
            return results
        prep = stage["result"]
        
        # Stage 2: Detection
        stage = benchmark_stage(
            "detection",
            detect_text_regions,
            prep["image"]
        )
        results["stages"]["detection"] = {
            **{k: v for k, v in stage.items() if k != "result"},
            "regions_found": len(stage["result"]) if stage["success"] else 0
        }
        if not stage["success"]:
            return results
        regions = stage["result"]
        
        # Stage 3: Recognition
        stage = benchmark_stage(
            "recognition",
            recognize_regions,
            regions
        )
        results["stages"]["recognition"] = {
            **{k: v for k, v in stage.items() if k != "result"},
            "regions_processed": len(stage["result"]) if stage["success"] else 0
        }
        if not stage["success"]:
            return results
        recognized = stage["result"]
        
        # Stage 4: Postprocessing
        stage = benchmark_stage(
            "postprocessing",
            process_recognition_results,
            recognized,
            currency
        )
        results["stages"]["postprocessing"] = {k: v for k, v in stage.items() if k != "result"}
        if not stage["success"]:
            return results
        processed = stage["result"]
        
        # Stage 5: Assembly
        stage = benchmark_stage(
            "assembly",
            assemble_menu,
            processed
        )
        results["stages"]["assembly"] = {
            **{k: v for k, v in stage.items() if k != "result"},
            "items": len(stage["result"]["items"]) if stage["success"] else 0,
            "orphans": len(stage["result"]["orphan_prices"]) if stage["success"] else 0
        }
        if not stage["success"]:
            return results
        
        # Calculate total time
        results["total_time_ms"] = sum(
            s.get("duration_ms", 0) for s in results["stages"].values()
        )
        results["success"] = True
        
        logger.info(f"Benchmark completed in {results['total_time_ms']:.1f}ms")
        
    except Exception as e:
        logger.exception("Benchmark failed")
        results["error"] = str(e)
    
    return results


def print_benchmark_results(results: Dict[str, Any]):
    """Print benchmark results in a readable format."""
    
    print("\n" + "="*70)
    print(f"BENCHMARK RESULTS: {Path(results['image_path']).name}")
    print("="*70)
    
    if not results["success"]:
        print(f"❌ FAILED: {results.get('error', 'Unknown error')}")
        return
    
    print(f"✅ Total Time: {results['total_time_ms']:.1f}ms ({results['total_time_ms']/1000:.2f}s)")
    print("\nPer-Stage Breakdown:")
    print("-"*70)
    
    for stage_name, stage_data in results["stages"].items():
        duration = stage_data.get("duration_ms", 0)
        percentage = (duration / results["total_time_ms"] * 100) if results["total_time_ms"] > 0 else 0
        
        status = "✅" if stage_data.get("success") else "❌"
        print(f"{status} {stage_name:15s}: {duration:7.1f}ms ({percentage:5.1f}%)", end="")
        
        # Add extra info for some stages
        if "regions_found" in stage_data:
            print(f" - {stage_data['regions_found']} regions", end="")
        if "items" in stage_data:
            print(f" - {stage_data['items']} items, {stage_data['orphans']} orphans", end="")
        
        print()
    
    # Memory summary
    print("\nMemory Usage:")
    print("-"*70)
    
    final_memory = list(results["stages"].values())[-1].get("memory_after", {})
    if "gpu_max_allocated_mb" in final_memory:
        print(f"GPU Peak: {final_memory['gpu_max_allocated_mb']:.1f} MB")
    if "cpu_ram_mb" in final_memory:
        print(f"CPU RAM: {final_memory['cpu_ram_mb']:.1f} MB")
    
    print("="*70)


def benchmark_multiple(image_paths: list, currency: str = DEFAULT_CURRENCY) -> Dict[str, Any]:
    """Benchmark multiple images and aggregate results."""
    
    all_results = []
    
    for path in image_paths:
        result = benchmark_full_pipeline(path, currency)
        all_results.append(result)
    
    # Calculate statistics
    successful = [r for r in all_results if r["success"]]
    
    if not successful:
        return {"error": "No successful benchmarks"}
    
    total_times = [r["total_time_ms"] for r in successful]
    
    summary = {
        "total_images": len(image_paths),
        "successful": len(successful),
        "failed": len(image_paths) - len(successful),
        "timing_stats": {
            "mean_ms": sum(total_times) / len(total_times),
            "min_ms": min(total_times),
            "max_ms": max(total_times),
        },
        "individual_results": all_results,
    }
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark menu scanner pipeline")
    parser.add_argument("images", nargs="+", help="Path(s) to menu image(s)")
    parser.add_argument("--currency", default=DEFAULT_CURRENCY, choices=["TND", "EUR"])
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(level="DEBUG" if args.verbose else "INFO")
    
    if len(args.images) == 1:
        # Single image benchmark
        results = benchmark_full_pipeline(args.images[0], args.currency)
        print_benchmark_results(results)
        
        if args.output:
            # Remove result objects before JSON serialization
            clean_results = {k: v for k, v in results.items() if k != "result"}
            Path(args.output).write_text(json.dumps(clean_results, indent=2))
            print(f"\n📊 Results saved to: {args.output}")
    
    else:
        # Multiple images benchmark
        print(f"\n🔄 Benchmarking {len(args.images)} images...\n")
        
        summary = benchmark_multiple(args.images, args.currency)
        
        # Print each individual result
        for result in summary["individual_results"]:
            print_benchmark_results(result)
            print()
        
        # Print summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Total Images: {summary['total_images']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        
        if summary["successful"] > 0:
            stats = summary["timing_stats"]
            print(f"\nTiming Statistics:")
            print(f"  Mean: {stats['mean_ms']:.1f}ms ({stats['mean_ms']/1000:.2f}s)")
            print(f"  Min:  {stats['min_ms']:.1f}ms ({stats['min_ms']/1000:.2f}s)")
            print(f"  Max:  {stats['max_ms']:.1f}ms ({stats['max_ms']/1000:.2f}s)")
        
        print("="*70)
        
        if args.output:
            Path(args.output).write_text(json.dumps(summary, indent=2))
            print(f"\n📊 Results saved to: {args.output}")
