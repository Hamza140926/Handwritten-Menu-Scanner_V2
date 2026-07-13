# Critical Action Items - Quick Reference
**Generated from:** code-analysis-report.md
**Priority:** Critical issues that must be addressed before production

---

## 🔴 CRITICAL - Fix Immediately (Before Any Production Use)

### 1. Thread Safety (2-3 days)
**Location:** `detection.py`, `recognition.py`
**Issue:** Global singletons without locks = race conditions in multi-threaded environments
```python
# Current (UNSAFE):
_detector = None
def _get_detector():
    global _detector
    if _detector is None:
        _detector = TextDetection(...)

# Fix: Add threading locks
import threading
_detector_lock = threading.Lock()
_detector = None

def _get_detector():
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = TextDetection(...)
    return _detector
```

### 2. Input Validation (1 week)
**Location:** All modules, especially `preprocessing.py`
**Issue:** No protection against malicious/malformed inputs

**Add:**
```python
# New file: src/validation.py
from pathlib import Path

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_DIMENSION = 10000
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

class ValidationError(Exception):
    pass

def validate_image_input(path: str) -> Path:
    path = Path(path).resolve()
    
    if not path.is_file():
        raise ValidationError("Path is not a valid file")
    
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValidationError(f"Unsupported format: {path.suffix}")
    
    size = path.stat().st_size
    if size > MAX_FILE_SIZE:
        raise ValidationError(f"File too large: {size} bytes (max {MAX_FILE_SIZE})")
    
    # Load and check dimensions
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise ValidationError("Could not load image")
    
    h, w = img.shape[:2]
    if h > MAX_DIMENSION or w > MAX_DIMENSION:
        raise ValidationError(f"Image dimensions too large: {w}x{h}")
    
    return path
```

### 3. Error Handling Framework (1 week)
**Location:** All modules
**Issue:** Exceptions bubble up uncaught, no graceful degradation

**Add:**
```python
# New file: src/exceptions.py
class PipelineError(Exception):
    """Base exception for pipeline errors"""
    pass

class PreprocessingError(PipelineError):
    pass

class DetectionError(PipelineError):
    pass

class RecognitionError(PipelineError):
    pass

class PostprocessingError(PipelineError):
    pass

# Update pipeline.py:
def run_pipeline_safe(image_path: str, default_currency: str = "TND") -> dict:
    """Safe pipeline execution with error recovery"""
    try:
        validate_image_input(image_path)
        return run_pipeline(image_path, default_currency)
    except ValidationError as e:
        return {"error": "validation_failed", "message": str(e), "items": []}
    except PreprocessingError as e:
        return {"error": "preprocessing_failed", "message": str(e), "items": []}
    except DetectionError as e:
        return {"error": "detection_failed", "message": str(e), "items": []}
    except RecognitionError as e:
        return {"error": "recognition_failed", "message": str(e), "items": []}
    except Exception as e:
        logger.exception("Unexpected pipeline failure")
        return {"error": "unknown", "message": "Internal error", "items": []}
```

### 4. Logging Infrastructure (1 week)

**Location:** All modules
**Issue:** Only print statements, no structured logging
**Add:**
```python
# New file: src/logging_config.py
import logging
import sys
from pathlib import Path

def setup_logging(log_file: Path = None, level: str = "INFO"):
    """Configure structured logging for the pipeline"""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.addHandler(console_handler)
    
    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

# In each module, add:
import logging
logger = logging.getLogger(__name__)

# Replace all print() with:
logger.info("Preprocessing started", extra={"image_path": path})
logger.warning("Skew detection fallback", extra={"reason": "no_lines_found"})
logger.error("Recognition failed", extra={"error": str(e)}, exc_info=True)
```

### 5. Dependency Pinning (1 day)
**Location:** `requirements.txt`

**Issue:** No version pinning = unpredictable behavior
**Fix:**
```bash
# Generate pinned requirements:
pip freeze > requirements-pinned.txt

# Or manually create requirements.txt with versions:
transformers==4.36.0
torch==2.1.0
pillow==10.1.0
opencv-python==4.8.1.78
numpy==1.26.2
sentencepiece==0.1.99
paddlepaddle==3.3.0
```

---

## 🟡 HIGH PRIORITY - Address Soon (1-2 Weeks)

### 6. Test Infrastructure (2-3 weeks)
**Location:** New `tests/` directory
**Issue:** Zero test coverage
**Create:**
```
tests/
  __init__.py
  conftest.py  # pytest fixtures
  test_preprocessing.py
  test_detection.py
  test_recognition.py
  test_postprocess.py
  test_pipeline.py
  integration/
    test_full_pipeline.py
  fixtures/
    sample_menus/  # Anonymized real menu crops
```

**Start with critical path tests:**
```python
# tests/test_postprocess.py
import pytest
from src.postprocess import extract_price

def test_price_extraction_decimal_dot():
    result = extract_price("12.50")
    assert result["value"] == 12.5
    assert result["raw"] == "12.50"
    assert not result["ambiguous"]

def test_price_extraction_decimal_comma():

    result = extract_price("12,50")
    assert result["value"] == 12.5
    assert result["raw"] == "12,50"

def test_price_ambiguous_multiple():
    result = extract_price("3, 3, 3.")
    assert result["ambiguous"] == True

def test_no_price():
    result = extract_price("Espresso")
    assert result["value"] is None
```

### 7. Configuration Management (2-3 days)
**Location:** New `src/config.py`
**Issue:** Magic numbers scattered everywhere
**Create:**
```python
# src/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PreprocessingConfig:
    max_dimension: int = 2000
    max_skew_degrees: float = 15.0
    denoise_strength: int = 7
    clahe_clip_limit: float = 2.5
    clahe_tile_size: tuple = (8, 8)

@dataclass
class DetectionConfig:
    min_box_area: int = 200
    model_name: str = "PP-OCRv5_mobile_det"
    device: str = "cpu"
    enable_mkldnn: bool = False

@dataclass
class RecognitionConfig:
    model_checkpoint: str = "microsoft/trocr-base-handwritten"

    batch_size: int = 8
    max_new_tokens: int = 32
    device: str = "auto"
    use_fp16_on_gpu: bool = True

@dataclass
class PostprocessConfig:
    price_regex: str = r"\b\d{1,4}(?:[.,]\d{1,3})?\b"
    min_reasonable_price: float = 0.01
    max_reasonable_price: float = 99999.0
    currency_confidence_threshold: float = 0.85
    default_currency: str = "TND"

@dataclass
class PipelineConfig:
    max_y_distance: float = 60
    column_gap_factor: float = 2.0
    relative_y_distance: bool = True  # Make y_distance relative to image height

@dataclass
class AppConfig:
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    detection: DetectionConfig = DetectionConfig()
    recognition: RecognitionConfig = RecognitionConfig()
    postprocess: PostprocessConfig = PostprocessConfig()
    pipeline: PipelineConfig = PipelineConfig()
    
    @classmethod
    def from_file(cls, path: Path):
        """Load config from YAML/JSON file"""
        # TODO: implement
        pass
```

### 8. Performance Benchmarking (1 week)
**Location:** New `benchmarks/` directory
**Issue:** Unknown performance characteristics

**Create:**
```python
# benchmarks/benchmark_pipeline.py
import time
import torch
from src.pipeline import run_pipeline

def benchmark_stage(name, func, *args, **kwargs):
    """Benchmark a single pipeline stage"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    duration = time.perf_counter() - start
    
    memory = None
    if torch.cuda.is_available():
        memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        torch.cuda.reset_peak_memory_stats()
    
    return {
        "stage": name,
        "duration_ms": duration * 1000,
        "gpu_memory_mb": memory,
        "result": result
    }

def benchmark_full_pipeline(image_path: str):
    """Benchmark entire pipeline with stage breakdowns"""
    stages = {}
    
    # Preprocessing
    from src.preprocessing import preprocess_image
    stages["preprocessing"] = benchmark_stage("preprocessing", preprocess_image, image_path)
    
    # Detection
    from src.detection import detect_text_regions
    prep = stages["preprocessing"]["result"]
    stages["detection"] = benchmark_stage("detection", detect_text_regions, prep["image"])
    
    # ... continue for all stages
    
    return stages

if __name__ == "__main__":
    results = benchmark_full_pipeline("test_menu.jpg")
    for stage, data in results.items():

        print(f"{stage}: {data['duration_ms']:.1f}ms, GPU: {data['gpu_memory_mb']:.1f}MB")
```

---

## 🟢 MEDIUM PRIORITY - Nice to Have (2-4 Weeks)

### 9. Edge Case Handling
- Empty detection results → return error flag
- All low confidence (<0.3) → flag entire menu
- Invalid price values (0.0, >99999) → bounds checking
- Add sanity checks to `pipeline.py` output

### 10. Better Price Regex
```python
# Current (too permissive):
PRICE_PATTERN = re.compile(r"\d+(?:[.,]\d{1,3})?")

# Better (word boundaries + validation):
PRICE_PATTERN = re.compile(r"\b\d{1,4}(?:[.,]\d{1,3})?\b")

def extract_price(text: str) -> dict:
    matches = PRICE_PATTERN.findall(text)
    # ... existing logic ...
    
    # Add bounds checking
    if value is not None:
        if value < MIN_REASONABLE_PRICE or value > MAX_REASONABLE_PRICE:
            # Flag as suspicious but don't discard
            return {
                "value": value, 
                "raw": chosen_raw, 
                "ambiguous": True,
                "out_of_bounds": True
            }
```

### 11. Type Hints Everywhere
```bash
# Add type hints to all functions
# Run mypy
pip install mypy
mypy src/ --strict
```

### 12. Memory Optimization

```python
# Add cleanup to pipeline
import gc

def run_pipeline(image_path: str, ...) -> dict:
    try:
        # ... pipeline code ...
        return result
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
```

### 13. Quality Metrics in Output
```python
# Update assemble_menu() to return quality metrics
def assemble_menu(processed_regions: list, ...) -> dict:
    # ... existing pairing logic ...
    
    # Calculate quality metrics
    total_items = len(items)
    paired_items = sum(1 for item in items if not item["is_category_header"])
    avg_confidence = sum(item["name_confidence"] for item in items) / max(total_items, 1)
    
    return {
        "items": items,
        "orphan_prices": orphan_prices,
        "quality_metrics": {
            "total_regions": len(processed_regions),
            "paired_items": paired_items,
            "orphan_prices": len(orphan_prices),
            "pairing_success_rate": paired_items / max(total_items, 1),
            "avg_confidence": avg_confidence,
            "low_confidence_count": sum(1 for item in items if item["name_confidence"] < 0.5)
        }
    }
```

---

## 📋 Checklist Format (Copy to Task Tracker)

```markdown
## Production Readiness Checklist

### Critical (Must Do)
- [ ] Add threading locks to singleton loaders (2-3 days)
- [ ] Implement input validation layer (1 week)

- [ ] Add error handling framework with custom exceptions (1 week)
- [ ] Set up structured logging infrastructure (1 week)
- [ ] Pin all dependency versions (1 day)

### High Priority (Should Do)
- [ ] Create test infrastructure with pytest (2-3 weeks)
  - [ ] Unit tests for each module
  - [ ] Integration tests for full pipeline
  - [ ] Fixtures with sample data
- [ ] Extract configuration to config.py (2-3 days)
- [ ] Benchmark performance characteristics (1 week)
- [ ] Optimize memory usage with cleanup (3-5 days)
- [ ] Add type hints throughout (1 week)

### Medium Priority (Nice to Have)
- [ ] Handle edge cases (empty detection, invalid prices) (1 week)
- [ ] Improve price regex with bounds checking (2-3 days)
- [ ] Add quality metrics to pipeline output (3-5 days)
- [ ] Create API documentation with Sphinx (3-5 days)
- [ ] Set up pre-commit hooks (black, flake8, mypy) (1 day)

### Future Enhancements
- [ ] Implement correction logging (spec §8) (2 weeks)
- [ ] Support for 3+ column layouts (1-2 weeks)
- [ ] RTL language support (2-3 weeks)
- [ ] Model serving infrastructure (2-3 weeks)

**Estimated Total Time to Production Ready:** 5-7 weeks
```

---

## 🚀 Quick Start: First Week Plan

**Day 1:**

- Morning: Pin all dependencies with exact versions
- Afternoon: Set up logging infrastructure
- End of day: Add threading locks to singletons

**Day 2-3:**
- Create validation.py with input validation
- Add ValidationError exception class
- Test validation with various inputs

**Day 4-5:**
- Create exceptions.py with custom exception hierarchy
- Wrap pipeline with error handling
- Add try/except blocks to each module
- Test error recovery

**End of Week 1:**
- Basic safety net in place
- Can start beta testing with supervision
- Foundation for week 2 test development

---

## 📞 Questions to Answer Before Starting

1. **Deployment Target:**
   - Will this run on-device or cloud server?
   - Mobile app or web app?
   - Single-user or multi-user concurrent?

2. **Performance Requirements:**
   - What's acceptable processing time? (target: <30 seconds?)
   - Memory constraints? (mobile: <500MB, server: flexible?)

3. **Data Privacy:**
   - Where are uploaded menus stored?
   - Retention policy?
   - Correction data collection consent?

4. **Integration:**
   - API contract with Scan2See app?
   - Data format expectations?
   - Error handling responsibilities?

**Action:** Schedule 1-hour meeting with product/architecture team to clarify these
