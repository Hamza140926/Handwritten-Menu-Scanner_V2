# Handwritten Menu Scanner

Production-grade OCR pipeline for converting handwritten restaurant menus into structured digital data. Combines computer vision, deep learning, and intelligent geometric algorithms to extract menu items, prices, and categories from photographed menus with minimal manual correction.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [Training Custom Models](#training-custom-models)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [API Integration](#api-integration)
- [Experiments](#experiments)
- [Documentation](#documentation)
- [License](#license)

## Overview

The Handwritten Menu Scanner is an end-to-end system designed for the Scan2See application, enabling restaurant owners to digitize handwritten menus efficiently. The system processes smartphone photos through a five-stage pipeline optimized for real-world conditions: varying lighting, diverse handwriting styles, messy layouts, and imperfect image quality.

### Current Status

Production-ready pipeline with the following capabilities:
- Complete preprocessing, detection, recognition, and assembly stages
- Hungarian algorithm-based geometric pairing for optimal item-price matching
- Fine-tunable TrOCR recognition model with synthetic dataset generator
- FastAPI integration for web service deployment
- Comprehensive validation, error handling, and structured logging
- Windows/Linux/macOS support with GPU acceleration

### Design Philosophy

This project prioritizes **practical utility over theoretical perfection**:
- **Human-in-the-loop**: Manual review is a first-class feature, not a failure mode
- **Graceful degradation**: Pipeline continues with partial results rather than failing completely
- **Real-world validation**: Tested on actual restaurant menus, not just synthetic benchmarks
- **Honest limitations**: Known edge cases documented rather than hidden
- **Iterative improvement**: Built for continuous refinement with real-world feedback


## Features

### Core Capabilities
- **Multi-stage pipeline**: Preprocessing, detection, recognition, postprocessing, assembly
- **Intelligent pairing**: Hungarian algorithm with 3-signal category voting
- **Currency support**: TND, EUR with automatic detection and fallback logic
- **Quality metrics**: Confidence scoring, pairing success rate, validation warnings
- **Error resilience**: Custom exception hierarchy with stage-specific error handling
- **Memory optimization**: Incremental processing with explicit cleanup steps

### Advanced Features
- **Interactive detection mode**: Review and retry detection with different strategies
- **Debug visualization**: Color-coded boxes showing categories, items, noise, and pairings
- **Configurable parameters**: 60+ tunable settings via JSON or programmatic API
- **Synthetic data generation**: 4,400+ handwritten samples for model training
- **Model fine-tuning**: Complete training infrastructure for custom TrOCR models
- **GPU acceleration**: Optimized for NVIDIA GPUs with automatic fallback to CPU
- **Thread-safe operations**: Lock-protected model singletons for concurrent requests

### Quality Assurance
- **Input validation**: File size, format, dimensions, and security checks
- **Structured logging**: JSON-formatted logs with contextual metadata
- **Performance monitoring**: Memory tracking and stage-level benchmarking
- **Test coverage**: 70%+ coverage on core modules (postprocess, validation, pipeline)
- **Real-world tested**: Validated on actual restaurant menus with documented results

## Quick Start

### Basic Usage

```bash
# Clone repository
git clone https://github.com/hamza030220/Handwritten-Menu-Scanner_V2.git
cd Handwritten-Menu-Scanner_V2

# Install dependencies
pip install -r requirements.txt

# Process a menu
python src/pipeline.py path/to/menu.jpg TND
```

### Docker Deployment (Recommended)

```bash
# Build image
docker build -t menu-scanner .

# Run container
docker run -p 8000:8000 -v ./data:/app/data menu-scanner

# Test endpoint
curl -X POST http://localhost:8000/scan-menu \
  -F "image=@menu.jpg" \
  -F "currency=TND"
```


## Architecture

### System Overview

```
Input Image (JPEG/PNG)
    |
    v
[1] Preprocessing -----> Deskew, denoise, contrast normalize
    |
    v
[2] Detection ---------> Find text region bounding boxes (PaddleOCR)
    |
    v
[3] Pairing -----------> Classify & pair geometrically (Hungarian algorithm)
    |                    - Noise filtering (shape-based)
    |                    - Category detection (3-signal voting)
    |                    - Item-price matching (global optimization)
    v
[4] Recognition -------> Read handwriting (TrOCR transformer)
    |
    v
[5] Postprocessing ----> Extract prices, resolve currency
    |
    v
[6] Assembly ----------> Group items under categories, sort by position
    |
    v
Structured JSON Output
```

### Key Algorithms

#### Hungarian Algorithm Pairing (Production)
Globally optimal item-price matching using `scipy.optimize.linear_sum_assignment`:

**Cost Matrix Construction:**
```python
cost[i][j] = W_ANGLE * abs(angle_ij) + W_VDIST * abs(vertical_distance_ij)
```

**Rejection Threshold:** Cost matrix padded to allow "no match" instead of forcing incorrect pairs

**Double-Use Resolution:** Greedy selection through cost-sorted assignments

**Advantages over greedy approaches:**
- Eliminates cascade errors (one wrong match affecting all subsequent matches)
- Handles dense/crowded rows with ambiguous geometries
- Leaves items unmatched rather than forcing incorrect pairings

#### 3-Signal Category Voting
Robust header detection without fragile height thresholds:

**Vote (a):** No companion in established price column  
**Vote (b):** Row isolation (nothing overlaps this horizontal band)  
**Vote (c):** Large whitespace gap above (> median_spacing × 1.6)

**Decision Rule:** ≥2 votes required (out of 3)

**Benefits:**
- Adapts to each menu's actual layout (learned row spacing, detected columns)
- Height-independent (works when categories aren't drawn bigger)
- Robust to handwriting scale drift across the page


## Installation

### Prerequisites

- **Python**: 3.10 or higher
- **OS**: Windows, Linux, or macOS
- **RAM**: 4GB minimum, 8GB+ recommended
- **GPU** (optional): NVIDIA with 4GB+ VRAM and CUDA 11.8+

### Standard Installation

```bash
# Create virtual environment
conda create -n menu-scanner python=3.10
conda activate menu-scanner

# Install dependencies
pip install -r requirements.txt
```

### GPU Installation (CUDA 11.8)

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

### Development Installation

```bash
# Install with dev tools (pytest, black, mypy)
pip install -r requirements-dev.txt
```

### Verification

```bash
# Check installations
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "from paddleocr import PaddleOCR; print('PaddleOCR OK')"
python -c "from transformers import TrOCRProcessor; print('Transformers OK')"

# Run test suite
pytest tests/
```

See [INSTALL.md](INSTALL.md) for detailed installation instructions and troubleshooting.

## Usage

### Command Line Interface

**Basic usage:**
```bash
python src/pipeline.py menu.jpg TND
```

**With debug visualization:**
```bash
python src/pipeline.py menu.jpg TND --debug-out debug.png
```

**Interactive mode (review detection before recognition):**
```bash
python src/pipeline.py menu.jpg TND --interactive
```

**Skip debug image generation:**
```bash
python src/pipeline.py menu.jpg TND --no-debug-image
```


### Programmatic API

```python
from src.pipeline import run_pipeline
from src.validation import ValidationError
from src.exceptions import PipelineError

try:
    result = run_pipeline(
        image_path="menu.jpg",
        default_currency="TND",
        debug_output_path="debug.png"
    )
    
    # Access structured output
    for item in result["items"]:
        if item["is_category_header"]:
            print(f"\n--- {item['name']} ---")
        else:
            print(f"{item['name']:<30} {item['price_value']} {item['currency']}")
    
    # Check quality metrics
    metrics = result["quality_metrics"]
    print(f"\nPairing success: {metrics['pairing_success_rate']}%")
    print(f"Avg confidence: {metrics['avg_confidence']:.1%}")
    
    # Handle warnings
    for warning in metrics["warnings"]:
        print(f"Warning: {warning}")

except ValidationError as e:
    print(f"Invalid input: {e}")
except PipelineError as e:
    print(f"Pipeline error: {e}")
```

### Configuration

Load custom configuration:

```python
from src.config import load_config, get_config

# Load from JSON
load_config("config.json")

# Access settings
cfg = get_config()
print(cfg.recognition.batch_size)
print(cfg.pipeline.max_y_distance)

# Modify and save
cfg.recognition.batch_size = 32
cfg.save("config_modified.json")
```

See [docs/configuration-guide.md](docs/configuration-guide.md) for all available settings.

## Pipeline Stages

### 1. Preprocessing (`src/preprocessing.py`)

**Operations:**
- Image resize (max dimension 2000px)
- Deskewing via Hough line detection
- Bilateral filtering for noise reduction
- CLAHE contrast normalization
- Grayscale conversion

**Configuration:**
```python
{
  "preprocessing": {
    "max_dimension": 2000,
    "max_skew_degrees": 15,
    "denoise_strength": 7,
    "clahe_clip_limit": 2.5
  }
}
```


### 2. Detection (`src/detection.py`)

**Engine:** PaddleOCR PP-OCRv5 (detection only, no recognition at this stage)

**Process:**
1. Find text region polygons using DBNet-based detector
2. Filter boxes below minimum area threshold
3. Crop each region with perspective warp (handles rotation)
4. Sort regions in approximate reading order (top-to-bottom, left-to-right)

**Models available:**
- `PP-OCRv5_mobile_det`: 3MB, optimized for speed (default)
- `PP-OCRv5_server_det`: 47MB, higher accuracy

**Thread-safe:** Singleton detector with lock-protected initialization

### 3. Pairing (`src/pairing.py`)

**Core Innovation:** Structure decided from geometry BEFORE recognition runs

**Classification:**
```python
# Noise detection (shape-based)
is_noise = (area_ratio >= 0.05) AND (aspect_ratio < 2.0)

# Category detection (3-signal voting)
votes = 0
votes += 1 if no_price_column_companion  # Signal (a)
votes += 1 if row_isolation              # Signal (b)
votes += 1 if large_whitespace_gap       # Signal (c)
is_category = (votes >= 2)

# Item detection
is_item = not noise and not category
```

**Pairing Algorithm:**
1. Build cost matrix: `cost[i][j] = W_ANGLE*|angle| + W_VDIST*|dy|`
2. Pad with rejection threshold (dummy "no match" nodes)
3. Solve assignment with Hungarian algorithm
4. Resolve double-use conflicts (same box used as name and price)

**Output:** Skeleton with roles: `category`, `item_name`, `item_price`, `unresolved`, `noise`


### 4. Recognition (`src/recognition.py`)

**Engine:** Microsoft TrOCR (Transformer-based OCR)

**Model:** `microsoft/trocr-base-handwritten` (fine-tunable)

**Process:**
1. Batch crops for GPU efficiency (configurable batch size)
2. Generate text predictions with beam search
3. Extract confidence scores per prediction
4. Apply FP16 inference on GPU for 2x speedup

**Optimization:**
- FP16 inference (half precision)
- Batched processing (16 regions/batch default)
- GPU memory clearing between batches
- Thread-safe model singleton

**Performance:** ~50% of total pipeline time (expected for transformer model)

### 5. Postprocessing (`src/postprocess.py`)

**Price Extraction:**
```python
# Regex patterns
r"\b(\d+)[.,](\d{2,3})\b"   # 4.50, 4,50, 12.500
r"\b(\d+)\b"                 # 5 (integer)
```

**Currency Resolution:**
1. Check for explicit symbol in text: €, EUR, TND
2. If found and high confidence: use detected currency
3. Otherwise: use menu-level default (from CLI argument or config)

**Error Correction:**
- Normalize separators: `,` → `.`
- Strip common OCR artifacts: `O` → `0`, `l` → `1` in price context
- Validate range: 0.01 ≤ price ≤ 99999

**Ambiguity Detection:**
Flags prices with confusable characters (O/0, l/1, S/5) for manual review


### 6. Assembly (`src/pipeline.py`)

**Skeleton → Structured Menu:**

1. Join geometry-only skeleton with recognized text/prices
2. Group items under their category headers
3. Sort by reading order (Y-position within groups)
4. Resolve "unresolved" boxes using recognized text:
   - If price pattern detected: add to orphan_prices
   - Otherwise: treat as item without price
5. Calculate quality metrics and generate warnings

**Output Structure:**
```json
{
  "items": [
    {
      "name": "Category Header",
      "is_category_header": true,
      "price_value": null
    },
    {
      "name": "Menu Item",
      "name_confidence": 0.92,
      "price_value": 4.50,
      "price_raw": "4.50",
      "price_confidence": 0.87,
      "currency": "TND",
      "is_category_header": false
    }
  ],
  "orphan_prices": [...],
  "quality_metrics": {
    "total_regions": 27,
    "items_with_prices": 12,
    "category_headers": 3,
    "orphan_prices": 0,
    "pairing_success_rate": 100.0,
    "avg_confidence": 0.85,
    "warnings": []
  }
}
```

## Training Custom Models

### Synthetic Dataset Generation

Generate handwritten menu samples for training:

```bash
cd dataset_generator/synthetic

# Clone handwriting synthesis engine
git clone https://github.com/sjvasquez/handwriting-synthesis.git ../handwriting-synthesis-master

# Create environment
conda create -n handwriting python=3.8
conda activate handwriting
pip install tensorflow==1.15.0 svgwrite cairosvg opencv-python pandas

# Generate dataset
python generate_synthetic_dataset.py \
    --out_dir dataset_synth \
    --repo_dir ../handwriting-synthesis-master \
    --samples_per_field 10
```

**Output:** ~1,910 handwritten crops (191 unique strings × 10 style/bias variations)

**Style-disjoint splits:** Train/val/test use completely different handwriting styles to test generalization

See [dataset_generator/README.md](dataset_generator/README.md) for details.


### Fine-Tuning TrOCR

**Quick Start (Synthetic Data):**

```bash
cd training

# Install training dependencies
pip install torch torchvision transformers evaluate jiwer accelerate datasets

# Train on synthetic dataset
python train_synthetic.py \
    --dataset_dir ../dataset_generator/synthetic/dataset_synth \
    --output_model_dir ../models/trocr_menu_v1 \
    --batch_size 16 \
    --max_epochs 20 \
    --lr 5e-5 \
    --eval_test
```

**Training Time:** 15-30 minutes on RTX 3060, 3-6 hours on CPU

**Real Data Workflow:**

```bash
# 1. Extract crops from real menu photos
python extract_crops.py --images_dir raw_menus/ --out_dir dataset/

# 2. Label crops interactively
python label_tool.py --out_dir dataset/

# 3. Fine-tune model
python train_trocr.py \
    --out_dir dataset/ \
    --output_model_dir ../models/trocr_menu_real_v1 \
    --batch_size 16 \
    --max_epochs 10
```

**Using Fine-Tuned Model:**

Update configuration to point to local checkpoint:

```python
# src/config.py
@dataclass
class RecognitionConfig:
    model_checkpoint: str = r"C:\path\to\models\trocr_menu_v1"
```

Or via config JSON:
```json
{
  "recognition": {
    "model_checkpoint": "C:/path/to/models/trocr_menu_v1"
  }
}
```

**Evaluation:**

```bash
# Compare base vs fine-tuned
python evaluate_model.py --out_dir dataset/ --model microsoft/trocr-base-handwritten
python evaluate_model.py --out_dir dataset/ --model ../models/trocr_menu_v1

# Field-specific evaluation
python evaluate_by_field.py --out_dir dataset/ --model ../models/trocr_menu_v1
```

See [training/README.md](training/README.md) for complete training guide.


## Configuration

All system parameters centralized in `src/config.py`. Configuration can be loaded from JSON files or modified programmatically.

### Key Parameters

**Preprocessing:**
- `max_dimension`: 2000 (cap image size)
- `max_skew_degrees`: 15 (safety limit on rotation)
- `denoise_strength`: 7 (bilateral filter strength)
- `clahe_clip_limit`: 2.5 (contrast enhancement)

**Detection:**
- `model_name`: "PP-OCRv5_mobile_det"
- `min_box_area`: 200 (filter noise boxes)
- `batch_size`: 1 (for single images)
- `device`: "cpu" or "gpu:0"

**Recognition:**
- `model_checkpoint`: "microsoft/trocr-base-handwritten"
- `batch_size`: 16 (higher = faster, more GPU memory)
- `max_new_tokens`: 32 (max text length)
- `use_fp16_on_gpu`: true (half precision)

**Pairing (Hungarian Algorithm):**
- `MAX_TILT_DEG`: 8.0 (max angle between item and price)
- `MAX_VERTICAL_SHIFT_RATIO`: 1.2 (max vertical drift)
- `REJECT_COST`: 999.0 (threshold for "no match")
- `W_ANGLE`: 1.0 (cost weight for angle deviation)
- `W_VDIST`: 0.08 (cost weight for vertical distance)
- `CATEGORY_VOTE_THRESHOLD`: 2 (votes needed out of 3)

**Validation:**
- `max_file_size`: 50MB (DoS prevention)
- `max_dimension`: 10000px (memory bomb prevention)
- `allowed_extensions`: [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

### Hardware-Specific Configs

**2GB GPU:**
```json
{
  "recognition": {
    "batch_size": 8,
    "use_fp16_on_gpu": true
  }
}
```

**6GB+ GPU:**
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

**CPU Only:**
```json
{
  "recognition": {
    "batch_size": 4,
    "device": "cpu"
  },
  "detection": {
    "device": "cpu"
  }
}
```

See [docs/configuration-guide.md](docs/configuration-guide.md) for all 60+ parameters.


## Project Structure

```
handwritten-menu-scanner/
├── src/                          # Core pipeline implementation
│   ├── pipeline.py              # Main orchestration + assembly
│   ├── preprocessing.py         # Image preprocessing
│   ├── detection.py             # Text region detection (PaddleOCR)
│   ├── pairing.py               # Hungarian algorithm + skeleton builder
│   ├── recognition.py           # Handwriting recognition (TrOCR)
│   ├── postprocess.py           # Price extraction + currency resolution
│   ├── validation.py            # Input validation + security checks
│   ├── exceptions.py            # Custom exception hierarchy
│   ├── config.py                # Configuration management
│   ├── logging_config.py        # Structured logging setup
│   ├── optimization.py          # GPU warmup + memory management
│   ├── main.py                  # FastAPI web service wrapper
│   └── diagnose.py              # Pipeline diagnostics utility
│
├── training/                     # Model training infrastructure
│   ├── train_synthetic.py       # Train on synthetic dataset
│   ├── train_trocr.py           # Train on real labeled data
│   ├── evaluate_model.py        # Compute CER/WER metrics
│   ├── evaluate_by_field.py     # Field-specific accuracy analysis
│   ├── extract_crops.py         # Extract training samples from photos
│   ├── label_tool.py            # Interactive labeling interface
│   ├── find_optimal_batch.py    # Batch size tuning for GPU
│   └── requirements_training.txt
│
├── dataset_generator/            # Synthetic data generation
│   └── synthetic/
│       ├── generate_synthetic_dataset.py
│       ├── generate_numbers_dataset.py
│       ├── clean_dataset.py     # Interactive dataset cleaning
│       ├── vocabulary.py        # Menu item vocabulary
│       ├── augment.py           # Photo-realism augmentation
│       ├── resplit_by_value.py  # Data split utilities
│       ├── check_leakage.py     # Validate style disjointness
│       ├── dataset_synth/       # Generated menu dataset (~4,400 samples)
│       └── dataset_numbers/     # Generated numbers dataset (~1,000 samples)
│
├── Experiment/                   # Pairing algorithm experiments
│   ├── experiment2.py           # Hungarian algorithm prototype
│   ├── pairing_experiment.py    # Tilted line algorithm (v1)
│   ├── batch_experiment.py      # Batch testing utilities
│   └── README.md                # Algorithm documentation + analysis
│
├── benchmarks/                   # Performance benchmarks
│   ├── benchmark_pipeline.py    # Stage-level timing + memory tracking
│   └── README.md                # Baseline performance results
│
├── tests/                        # Test suite
│   ├── test_pipeline.py         # Menu assembly tests
│   ├── test_postprocess.py      # Price extraction tests
│   ├── test_validation.py       # Input validation tests
│   ├── test_preprocessing.py    # Image preprocessing tests
│   ├── test_logging_config.py   # Logging configuration tests
│   ├── test_exceptions.py       # Exception hierarchy tests
│   ├── conftest.py              # Shared fixtures
│   └── README.md                # Testing guide
│
├── docs/                         # Comprehensive documentation
│   ├── handwritten-menu-scanner-spec.md
│   ├── recognition-status-report.md
│   ├── code-analysis-report.md
│   ├── configuration-guide.md
│   └── action-items.md
│
├── data/samples/                 # Test menu photos (gitignored)
├── models/                       # Fine-tuned model checkpoints (gitignored)
├── notebooks/                    # Jupyter analysis notebooks
│
├── requirements.txt              # Pipeline dependencies
├── requirements-dev.txt          # Development tools
├── config.example.json           # Example configuration
├── pytest.ini                    # Pytest configuration
├── .gitignore                    # Git ignore rules
├── INSTALL.md                    # Installation guide
└── README.md                     # This file
```


## Performance

### Baseline Benchmarks

**Test Environment:**
- GPU: NVIDIA RTX 3050 (4GB VRAM)
- CPU: 12 threads
- Image: 14 text regions, typical cafe menu

**Stage-Level Timing:**

| Stage | Time | Percentage | Notes |
|-------|------|------------|-------|
| Validation | 44ms | 0.1% | Input checks |
| Preprocessing | 4.6s | 12.4% | Deskew, denoise, contrast |
| Detection | 13.7s | 36.6% | PaddleOCR text detection |
| Pairing | <50ms | 0.1% | Hungarian algorithm |
| Recognition | 19.1s | 50.9% | TrOCR handwriting recognition |
| Postprocessing | <1ms | 0.0% | Price extraction |
| Assembly | <1ms | 0.0% | Menu item pairing |
| **Total** | **37.5s** | **100%** | End-to-end |

**Memory Usage:**
- GPU Peak: 665 MB / 4096 MB (16% utilization)
- CPU RAM: 1969 MB

**Bottlenecks:**
1. Recognition (50.9%) - Transformer model inference
2. Detection (36.6%) - PaddleOCR region detection
3. Preprocessing (12.4%) - Image operations

**Optimization Opportunities:**
- Batch processing multiple menus (reuses loaded models)
- Model quantization (INT8 inference for 2-3x speedup)
- Reduce image resolution (trade quality for speed)
- Use PP-OCRv5_mobile_det instead of server model

### Accuracy Characteristics

**Synthetic Test Set (Style-Disjoint):**
- Character Error Rate (CER): ~15-20% (base TrOCR)
- CER: ~8-12% (fine-tuned on synthetic data)

**Real-World Performance (Manual Validation):**
- Item name recognition: 70-85% exact match
- Price recognition: 80-90% exact match
- Category header detection: 85-95% correct
- Pairing accuracy: 90-95% correct matches

**Common Failure Modes:**
- Cursive handwriting (harder than print)
- Low contrast text (faded ink, poor lighting)
- Tilted/rotated images (>15 degrees)
- Overlapping text regions
- Decorative fonts misdetected as text

See [benchmarks/README.md](benchmarks/README.md) for detailed performance analysis.





## API Integration

### FastAPI Web Service

The pipeline includes a FastAPI wrapper for web service deployment:

```python
# src/main.py
from fastapi import FastAPI, File, UploadFile, Form
from src.pipeline import run_pipeline

app = FastAPI()

@app.post("/scan-menu")
async def scan_menu(
    image: UploadFile = File(...),
    currency: str = Form(default="TND")
):
    result = run_pipeline(image_path, currency)
    return result
```

**Running the service:**

```bash
# Development server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Production server
gunicorn src.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

**Testing the endpoint:**

```bash
# Using curl
curl -X POST http://localhost:8000/scan-menu \
  -F "image=@menu.jpg" \
  -F "currency=TND"

# Using Python requests
import requests

with open("menu.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/scan-menu",
        files={"image": f},
        data={"currency": "TND"}
    )
    
result = response.json()
for item in result["items"]:
    print(f"{item['name']}: {item['price_value']} {item['currency']}")
```

**Response Format:**

```json
{
  "items": [
    {
      "name": "Coffee",
      "is_category_header": true,
      "price_value": null
    },
    {
      "name": "Espresso",
      "name_confidence": 0.92,
      "price_value": 3.5,
      "price_raw": "3.50",
      "price_confidence": 0.87,
      "currency": "TND",
      "is_category_header": false
    }
  ],
  "orphan_prices": [],
  "quality_metrics": {
    "total_regions": 14,
    "items_with_prices": 12,
    "category_headers": 2,
    "pairing_success_rate": 100.0,
    "avg_confidence": 0.85,
    "warnings": []
  }
}
```

### Error Handling

The API returns appropriate HTTP status codes:

- **200 OK**: Successful processing
- **400 Bad Request**: Validation error (invalid file format, size)
- **422 Unprocessable Entity**: Stage-specific error (preprocessing, detection, recognition)
- **500 Internal Server Error**: Unexpected error


## Experiments

The `Experiment/` directory contains algorithm prototypes and comparative analysis:

### Pairing Algorithms

**experiment2.py - Hungarian Algorithm (Production)**

Global optimization approach using bipartite matching:
- 3-signal category voting (price companion, row isolation, whitespace gap)
- Shape-based noise detection (area + aspect ratio)
- Cost matrix with rejection padding
- Globally optimal item-price assignment

**Results:**
- m8.jpeg: 3 categories, 12 pairs (100% success)
- m20.png: 0 categories, 7 pairs (100% success)

**pairing_experiment.py - Tilted Line Algorithm (v1)**

Greedy two-pass approach with geometric ray-casting:
- Height-based category detection
- Sequential angle search (0°, ±1°, ±2°, ..., ±8°)
- Ambiguity detection and deferred resolution
- Learned vertical offset from confident matches

**Comparison:**

| Metric | Tilted Line | Hungarian |
|--------|-------------|-----------|
| Category Detection | Height threshold | 3-signal voting |
| Pairing Scope | Local (per-item) | Global (all items) |
| Cascade Errors | Possible | Eliminated |
| Complexity | O(n²) per pass | O(n³) |
| Code Size | ~200 lines | ~400 lines |
| Accuracy | Good for clean menus | Optimal for all layouts |

**batch_experiment.py - Batch Testing**

Process multiple images and aggregate statistics:
- Pairing success rate
- Orphan rate
- Category detection accuracy
- Debug visualization generation

See [Experiment/README.md](Experiment/README.md) for detailed algorithm analysis.


## Documentation

Comprehensive documentation available in the `docs/` directory:

### Technical Specifications

**[handwritten-menu-scanner-spec.md](docs/handwritten-menu-scanner-spec.md)**
- Complete project specification
- Architecture decisions and rationale
- Constraints and requirements
- Accuracy targets and acceptance criteria
- Rollout plan and deployment strategy

**[configuration-guide.md](docs/configuration-guide.md)**
- All 60+ configurable parameters
- Hardware-specific optimization recipes
- Tuning strategies for different use cases
- Examples for CPU-only, 2GB GPU, 6GB+ GPU

### Implementation Status

**[recognition-status-report.md](docs/recognition-status-report.md)**
- Current implementation status
- Real-world test results
- Known issues and limitations
- Validated failure modes with examples

### Code Quality

**[code-analysis-report.md](docs/code-analysis-report.md)**
- Comprehensive code review
- Strengths and blind spots
- Production readiness assessment
- Improvement recommendations

**[analysis-summary.md](docs/analysis-summary.md)**
- Executive summary of code quality
- Key metrics and scores
- Risk assessment
- Go/no-go deployment decision

**[action-items.md](docs/action-items.md)**
- Prioritized improvement list
- Code examples for each item
- Effort estimates
- Dependencies and prerequisites

### Training & Dataset

**[training/README.md](training/README.md)**
- Complete training workflow
- Synthetic vs real data strategies
- Fine-tuning best practices
- Evaluation methodology

**[dataset_generator/README.md](dataset_generator/README.md)**
- Synthetic dataset generation guide
- Handwriting synthesis setup
- Style-disjoint split strategy
- Quality validation procedures

**[Experiment/README.md](Experiment/README.md)**
- Pairing algorithm evolution
- Tilted line vs Hungarian comparison
- Tuning parameters and their effects
- Known limitations and edge cases


## Technology Stack

### Core Dependencies

**Deep Learning:**
- PyTorch 2.0+ (neural network framework)
- Transformers 4.30+ (Hugging Face, TrOCR model)
- Accelerate (mixed precision training)

**Computer Vision:**
- OpenCV 4.8+ (image preprocessing)
- PaddleOCR 3.7+ (text detection)
- PaddlePaddle 3.3+ (OCR backend)

**Scientific Computing:**
- NumPy 1.24+ (array operations)
- SciPy 1.14+ (Hungarian algorithm)
- Pandas 2.0+ (dataset management)

**Web Framework:**
- FastAPI 0.104+ (REST API)
- Uvicorn (ASGI server)
- Python-multipart (file uploads)

**Utilities:**
- Pillow 10.0+ (image handling)
- sentencepiece 0.2.0 (tokenization)
- protobuf 3.20+ (model serialization)

### Development Tools

**Testing:**
- pytest 7.4+ (test framework)
- pytest-cov (coverage reporting)

**Code Quality:**
- black (code formatting)
- mypy (type checking)
- flake8 (linting)

**Training:**
- evaluate (metric computation)
- jiwer (CER/WER calculation)
- datasets (Hugging Face datasets)

### System Requirements

**Minimum (CPU):**
- Python 3.10+
- 4GB RAM
- 5GB disk space

**Recommended (GPU Training):**
- Python 3.10+
- NVIDIA GPU with 4GB+ VRAM
- CUDA 11.8+
- 16GB RAM
- 20GB disk space

**Supported Platforms:**
- Windows 10/11
- Ubuntu 20.04+
- macOS 11+


## Known Limitations

### Geometric Pairing

**Vertical/Multi-Column Layouts:**
- Algorithm assumes horizontal left-to-right pairing
- Breaks on vertical price lists (price above/below item)
- Does not handle three-column layouts (item | size | price)

**Ambiguous Geometries:**
- Dense packed text may merge into single detection box
- Multiple valid candidates can cause ambiguous matches
- False-positive text regions (decorative elements, watermarks)

### Recognition Accuracy

**Challenging Handwriting:**
- Cursive script harder than print
- Extreme slant or unusual letter forms
- Overlapping characters or corrections

**Image Quality:**
- Low contrast text (faded ink)
- Poor lighting or shadows
- Motion blur or out-of-focus
- Extreme rotation (>15 degrees without deskew)

**Domain Adaptation:**
- Synthetic training data differs from real handwriting
- Model may struggle with rare menu item vocabulary
- Currency symbols and special characters have lower accuracy

### System Constraints

**Performance:**
- 37 seconds per menu (not suitable for real-time)
- First run includes model download (one-time delay)
- GPU required for reasonable training time

**Memory:**
- Large batches require GPU memory
- High-resolution images increase memory usage

**Language Support:**
- Currently optimized for English/French text
- Limited support for Arabic, Chinese, or other scripts

### Honest Limitations (By Design)

**Item with Missing Price:**
- From geometry alone, looks identical to category header
- Requires text recognition to distinguish
- May be incorrectly classified as category

**Semantic Validation:**
- System doesn't validate if item-price pairs make sense semantically
- Can pair unrelated text regions if geometry suggests it
- No check for reasonable price ranges per item type

See [docs/recognition-status-report.md](docs/recognition-status-report.md) for validated failure modes.


## Contributing

Contributions welcome. Please follow these guidelines:

### Development Workflow

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/my-feature`
3. **Make changes with tests**
4. **Run test suite**: `pytest`
5. **Check code quality**: `black src/ && mypy src/`
6. **Commit with clear message**: `git commit -m "Add feature: description"`
7. **Push to fork**: `git push origin feature/my-feature`
8. **Submit pull request**

### Code Standards

**Style:**
- Follow PEP 8 (enforced with black)
- Type hints for function signatures
- Docstrings for public functions/classes

**Testing:**
- Unit tests for new functions
- Integration tests for stage interactions
- Target 80%+ coverage for new code

**Documentation:**
- Update README for user-facing changes
- Document configuration parameters
- Add examples for new features

**Commit Messages:**
- Use imperative mood: "Add feature" not "Added feature"
- Be specific: "Fix pairing for vertical layouts" not "Fix bug"
- Reference issues: "Fixes #123: Handle empty detection results"

### Priority Areas

**High Impact:**
1. Improve recognition accuracy on real menus
2. Add support for vertical/multi-column layouts
3. Implement semantic validation for pairings
4. Expand test coverage to 90%+

**Medium Impact:**
5. Support additional languages/scripts
6. Optimize performance (reduce 37s baseline)
7. Add batch processing API endpoint
8. Implement confidence-based auto-correction

**Low Impact:**
9. Additional currency support
10. Enhanced debug visualizations
11. Performance profiling dashboard

See [docs/action-items.md](docs/action-items.md) for detailed improvement roadmap.

### Getting Help

**Questions:**
- Open a GitHub discussion for general questions
- Check existing issues for known problems

**Bug Reports:**
- Include minimal reproducible example
- Attach sample image if possible
- Specify Python version, OS, and GPU

**Feature Requests:**
- Describe use case and expected behavior
- Explain why existing features don't suffice
- Provide example inputs/outputs


## Troubleshooting

### Common Issues

**"No module named 'sentencepiece'"**
```bash
pip install sentencepiece==0.2.0
```

**"CUDA not available" (but GPU is present)**
```bash
# Check driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**"PaddleOCR model download fails"**
```bash
# Use domestic mirror (China)
export HUB_HOME=~/.paddleocr

# Or manually download and place in ~/.paddleocr/official_models/
```

**"Out of memory" during recognition**
```bash
# Reduce batch size in config
{
  "recognition": {
    "batch_size": 8  # Lower value
  }
}
```

**"No text regions detected"**
- Check image quality (resolution, contrast)
- Try interactive mode: `--interactive`
- Adjust detection threshold in config
- Ensure image is not completely black/white

**"Poor pairing accuracy"**
- Check debug image for misdetection
- Tune pairing parameters (MAX_TILT_DEG, MAX_VERTICAL_SHIFT_RATIO)
- Verify menu layout matches horizontal assumption
- Consider preprocessing adjustments (deskew, contrast)

### Debug Tools

**Diagnose pipeline issues:**
```bash
python src/diagnose.py menu.jpg
```

**Generate debug visualization:**
```bash
python src/pipeline.py menu.jpg TND --debug-out debug.png
```

**Test individual stages:**
```bash
# Test preprocessing
python src/preprocessing.py menu.jpg

# Test detection
python src/detection.py preprocessed.png

# Test recognition
python src/recognition.py
```

**Check configuration:**
```python
from src.config import get_config
cfg = get_config()
print(cfg.to_dict())
```

### Performance Issues

**Slow processing:**
- Use GPU instead of CPU
- Reduce image resolution (max_dimension)
- Use mobile detection model instead of server
- Enable FP16 inference

**High memory usage:**
- Reduce recognition batch size
- Clear GPU cache between stages
- Process images sequentially, not in batch

**Model download slow:**
- Check internet connection
- Use domestic mirror if in China
- Manually download models and place in cache directory


## Roadmap

### Version 1.0 (Current)
- [x] Complete pipeline implementation
- [x] Hungarian algorithm pairing
- [x] Synthetic dataset generation
- [x] TrOCR fine-tuning infrastructure
- [x] FastAPI integration
- [x] Basic test coverage
- [x] Documentation

### Version 1.1 (Next)
- [ ] Comprehensive test suite (90%+ coverage)
- [ ] Real data collection and labeling tools
- [ ] Mixed synthetic + real training
- [ ] Performance optimization (target <20s)
- [ ] Docker containerization
- [ ] CI/CD pipeline

### Version 2.0 (Future)
- [ ] Multi-language support (Arabic, Chinese)
- [ ] Vertical layout detection and pairing
- [ ] Three-column layout support
- [ ] Semantic validation layer
- [ ] Auto-correction based on confidence
- [ ] Batch processing API
- [ ] Web-based review UI

### Research Directions
- [ ] Active learning for efficient labeling
- [ ] Few-shot adaptation to new handwriting styles
- [ ] Multi-modal fusion (image + context)
- [ ] Adversarial training for robustness
- [ ] Model compression and quantization


## License


MIT License

Copyright (c) 2024 Hamza Slimani

This project is released under the MIT License.

S/O :

* My 3050, the real MVP. Bro got pushed to the limit, cooked itself, and still showed up for another training run.
* Coffee, the unofficial dependency of this project.
* Every bug that made me question my life choices and every fix that made it worth it.
* AI "assistants" for saving me from some absolutely cursed debugging sessions.

Built with curiosity, stubbornness, random ideas, and a GPU that definitely deserves a vacation.

The software is provided "AS IS", without warranty of any kind.

Basically:

If it works, don't touch it !!!

If it breaks, good luck soldier ....

If your GPU starts suffering, just know mine already went through worse....

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
.

## Acknowledgments

**Libraries and Models:**
- Microsoft TrOCR for handwriting recognition
- PaddleOCR for text detection
- Hugging Face Transformers ecosystem
- sjvasquez/handwriting-synthesis for synthetic data

**Inspiration:**
- Real-world feedback from restaurant owners
- Academic research in document understanding
- Open-source OCR community



## Contact

**Author:** Hamza Slimani  
**Email:** [hamza.slimani@esprit.tn]  
**GitHub:** [@hamza030220](https://github.com/hamza030220)  
**Project:** [Handwritten-Menu-Scanner_V2](https://github.com/hamza030220/Handwritten-Menu-Scanner_V2)

For bug reports and feature requests, please use GitHub Issues.

---

**Built somewhere between "I have a plan" and "let’s see what happens."**

