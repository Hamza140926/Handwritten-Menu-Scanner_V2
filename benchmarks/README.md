# Performance Benchmarks

Results from benchmarking the handwritten menu scanner pipeline.

## Running Benchmarks

```bash
# Single image
python benchmarks/benchmark_pipeline.py path/to/menu.jpg

# Multiple images
python benchmarks/benchmark_pipeline.py menu1.jpg menu2.jpg menu3.jpg

# Save results to JSON
python benchmarks/benchmark_pipeline.py menu.jpg --output results.json

# Verbose logging
python benchmarks/benchmark_pipeline.py menu.jpg --verbose
```

## Baseline Performance

**Test Environment:**
- GPU: RTX 3050 (4GB VRAM)
- CPU: 12 threads
- Image: 14 text regions, typical cafe menu

**Results:**

| Stage | Time | Percentage | Notes |
|-------|------|------------|-------|
| Validation | 44ms | 0.1% | Input checks |
| Preprocessing | 4.6s | 12.4% | Deskew, denoise, contrast |
| Detection | 13.7s | 36.6% | PaddleOCR text detection |
| Recognition | 19.1s | 50.9% | TrOCR handwriting recognition |
| Postprocessing | <1ms | 0.0% | Price extraction |
| Assembly | <1ms | 0.0% | Menu item pairing |
| **TOTAL** | **37.5s** | **100%** | End-to-end |

**Memory Usage:**
- GPU Peak: 665 MB
- CPU RAM: 1969 MB

## Performance Characteristics

**Bottlenecks:**
1. Recognition (50.9%) - TrOCR transformer model
2. Detection (36.6%) - PaddleOCR region detection
3. Preprocessing (12.4%) - Image operations

**Observations:**
- Recognition and detection are compute-heavy (expected for ML models)
- Fast stages (<1ms): postprocessing, assembly
- Memory usage well under limits (4GB GPU available)

## Performance Targets

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Total Time | 37.5s | <60s | ✅ Acceptable |
| GPU Memory | 665MB | <4GB | ✅ Good |
| Success Rate | 100% | >95% | ✅ Good |

## Optimization Opportunities

If faster processing is needed:

**High Impact:**
1. **Batch Processing** - Process multiple menus in one run (reuses loaded models)
2. **Async Processing** - Process in background, don't block user
3. **Cache Results** - Don't reprocess same menu twice

**Medium Impact:**
4. **GPU Optimization** - Increase batch size for recognition
5. **Model Quantization** - Use INT8 instead of FP16 (2-3x faster, slight accuracy loss)
6. **Smaller Detection Model** - Already using mobile version

**Low Impact:**
7. **Reduce Image Resolution** - Trade quality for speed
8. **Skip Preprocessing Steps** - If input is already good quality

## Notes

- 37 seconds is acceptable for batch/offline processing
- Not suitable for real-time camera processing (would need <5s)
- First run includes model download time (one-time cost)
- Subsequent runs use cached models
