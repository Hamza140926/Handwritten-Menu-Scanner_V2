# Configuration Guide

All system parameters are now centralized in `src/config.py` and can be tuned without code changes.

## Quick Start

The system works with default values out of the box. To customize:

1. Copy the example config:
   ```
   cp config.example.json config.json
   ```

2. Edit `config.json` with your values

3. Load it in your code:
   ```python
   from config import load_config
   load_config("config.json")
   ```

## Configuration Sections

### Preprocessing
- `max_dimension`: 2000px - cap image size for consistent processing
- `max_skew_degrees`: 15° - safety limit on rotation correction
- `denoise_strength`: 7 - higher = more denoising, more blur
- `clahe_clip_limit`: 2.5 - contrast enhancement strength

### Detection
- `model_name`: "PP-OCRv5_mobile_det" or "PP-OCRv5_server_det"
- `min_box_area`: 200px² - filter out noise
- `batch_size`: 1 - for single images
- `device`: "cpu" or "gpu:0"
- `row_tolerance`: 20px - regions within this are same row

### Recognition
- `model_checkpoint`: "microsoft/trocr-base-handwritten"
- `batch_size`: 16 - higher = faster but more GPU memory
- `max_new_tokens`: 32 - max text length
- `device`: "auto", "cpu", or "cuda"
- `use_fp16_on_gpu`: true - half precision (faster, less memory)

### Postprocessing
- `price_pattern`: regex for price extraction
- `min_reasonable_price`: 0.01 - minimum valid price
- `max_reasonable_price`: 99999.0 - maximum valid price
- `currency_confidence_threshold`: 0.85 - min confidence for detected currency
- `default_currency`: "TND" - default menu currency
- `supported_currencies`: ["TND", "EUR"]

### Pipeline
- `column_gap_factor`: 2.0 - gap size to split columns
- `max_y_distance`: 60px - max vertical distance to pair name with price
- `min_confidence_warning`: 0.5 - warn if confidence below this
- `max_orphan_price_ratio`: 0.3 - warn if >30% prices orphaned

### Validation
- `max_file_size`: 50MB - prevent DoS
- `min_file_size`: 1KB - prevent corrupt files
- `max_dimension`: 10000px - prevent memory bombs
- `min_dimension`: 100px - too small to be real menu
- `allowed_extensions`: [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]

### Optimization
- `enable_tf32`: true - TensorFloat-32 on Ampere GPUs (2x speedup)
- `enable_cudnn_benchmark`: true - auto-tune convolution algorithms
- `clear_cache_after_batch`: true - free GPU memory between batches
- `warmup_gpu_on_startup`: true - avoid first-run slowness

## Tuning for Your Hardware

### 2GB GPU
```json
{
  "recognition": {
    "batch_size": 8,
    "use_fp16_on_gpu": true
  }
}
```

### 6GB+ GPU
```json
{
  "recognition": {
    "batch_size": 24
  },
  "detection": {
    "model_name": "PP-OCRv5_server_det"
  }
}
```

### CPU Only
```json
{
  "recognition": {
    "batch_size": 4,
    "device": "cpu"
  },
  "detection": {
    "device": "cpu"
  },
  "optimization": {
    "warmup_gpu_on_startup": false
  }
}
```

## Programmatic Access

```python
from config import get_config, set_config, Config

# Get current config
cfg = get_config()
print(cfg.preprocessing.max_dimension)

# Modify and set
cfg.recognition.batch_size = 32
set_config(cfg)

# Load from file
from config import load_config
load_config("my_config.json")

# Save current config
cfg.save("backup_config.json")

# Reset to defaults
from config import reset_config
reset_config()
```
