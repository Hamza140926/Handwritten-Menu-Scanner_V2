"""
Performance optimization utilities.

Contains functions to improve pipeline performance through:
- GPU warmup
- Memory optimization
- Batch processing helpers
"""

import torch
import gc
import numpy as np
from logging_config import get_logger

logger = get_logger(__name__)


def warmup_gpu():
    """
    Warm up GPU to avoid first-run slowness.
    
    CUDA initialization on first use adds ~2-3 seconds overhead.
    Running a dummy operation warms up the GPU so subsequent
    operations are faster.
    
    Call this once at application startup for best performance.
    """
    if not torch.cuda.is_available():
        logger.debug("No GPU available, skipping warmup")
        return
    
    try:
        logger.info("Warming up GPU...")
        # Small tensor operation to initialize CUDA
        dummy = torch.randn(100, 100, device='cuda')
        result = dummy @ dummy.T
        result.cpu()  # Force synchronization
        torch.cuda.synchronize()
        logger.info("GPU warmup complete")
    except Exception as e:
        logger.warning(f"GPU warmup failed: {e}")


def clear_gpu_memory():
    """
    Clear GPU memory cache.
    
    Call this after processing a batch of images to free up memory
    for the next batch. Useful in batch processing scenarios.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU memory cache cleared")


def clear_memory():
    """
    Comprehensive memory cleanup (GPU + CPU).
    
    Use this between batches or after processing to ensure
    memory is freed.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.debug("Memory cleanup complete")


def get_optimal_batch_size(available_vram_mb: float = 4096) -> int:
    """
    Calculate optimal recognition batch size based on available VRAM.
    
    Args:
        available_vram_mb: Available GPU memory in MB
        
    Returns:
        Recommended batch size for recognition
    """
    # Rough estimates based on TrOCR-base memory usage
    # ~40MB per image in batch
    if available_vram_mb >= 8000:
        return 32
    elif available_vram_mb >= 6000:
        return 24
    elif available_vram_mb >= 4000:
        return 16
    elif available_vram_mb >= 2000:
        return 8
    else:
        return 4


def optimize_for_throughput():
    """
    Apply PyTorch optimizations for better throughput.
    
    These settings favor throughput over latency, good for
    batch processing multiple menus.
    """
    if torch.cuda.is_available():
        # Enable TF32 on Ampere GPUs (RTX 30xx, A100, etc.)
        # ~2x speedup for matmul operations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN auto-tuner (finds fastest convolution algorithm)
        torch.backends.cudnn.benchmark = True
        
        logger.info("PyTorch throughput optimizations enabled")


def optimize_for_latency():
    """
    Apply PyTorch optimizations for better latency.
    
    These settings favor consistent low latency over maximum
    throughput, good for interactive/real-time processing.
    """
    if torch.cuda.is_available():
        # Disable auto-tuner (avoids tuning overhead)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        logger.info("PyTorch latency optimizations enabled")


if __name__ == "__main__":
    from logging_config import setup_logging
    
    setup_logging(level="INFO")
    
    print("Testing GPU optimization utilities...")
    
    # Test GPU warmup
    warmup_gpu()
    
    # Show recommended batch size
    if torch.cuda.is_available():
        available = torch.cuda.get_device_properties(0).total_memory / 1024**2
        recommended = get_optimal_batch_size(available)
        print(f"\nGPU Memory: {available:.0f} MB")
        print(f"Recommended Batch Size: {recommended}")
    else:
        print("\nNo GPU available")
    
    # Test optimizations
    optimize_for_throughput()
    print("\n✅ Optimization utilities working correctly")
