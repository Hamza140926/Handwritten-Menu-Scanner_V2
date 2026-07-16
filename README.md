# Handwritten Menu Scanner

An AI-powered pipeline that converts handwritten paper menus into structured, editable digital formats. Built for the Scan2See application, this system helps restaurant and cafe owners digitize their menus without manual data entry.

## Overview

The scanner processes photos of handwritten menus through a multi-stage pipeline: image preprocessing, text detection, handwriting recognition, and intelligent item-price pairing. The system is designed for real-world constraints—general handwriting styles, messy layouts, and varying lighting conditions—achieving practical accuracy suitable for human review and correction.

**Current Status:** Complete pipeline with TrOCR fine-tuning capability. Includes synthetic dataset generator and optimized training infrastructure for custom model development.

## Quick Start

### Basic Usage

```bash
# Clone the repository
git clone https://github.com/hamza030220/Handwritten-Menu-Scanner_V2.git
cd Handwritten-Menu-Scanner_V2

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python src/pipeline.py path/to/menu_photo.jpg
```

### Training Custom Model

```bash
# Create training environment
conda create -n training python=3.10 -y
conda activate training

# Install training dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers pillow pandas numpy evaluate jiwer accelerate sentencepiece protobuf datasets

# Train on synthetic dataset (3-4 hours on RTX 3050)
cd training
python train_synthetic.py \
    --dataset_dir ../dataset_generator/synthetic/dataset_synth \
    --output_model_dir ../models/trocr_menu_v1 \
    --batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_epochs 5 \
    --lr 5e-5 \
    --eval_test
```

For detailed installation instructions, including GPU setup, see [INSTALL.md](INSTALL.md).

## Architecture

The pipeline consists of five stages:

1. **Preprocessing** (`src/preprocessing.py`) — Deskewing, denoising, contrast normalization
2. **Detection** (`src/detection.py`) — Text region detection using PaddleOCR
3. **Recognition** (`src/recognition.py`) — Handwriting recognition with Microsoft TrOCR
4. **Postprocessing** (`src/postprocess.py`) — Price extraction and currency resolution
5. **Assembly** (`src/pipeline.py`) — Intelligent pairing of item names with prices

Each stage is independently testable with clear input/output contracts.

## Project Structure

```
├── src/                      # Core pipeline implementation
│   ├── pipeline.py          # Main orchestration
│   ├── preprocessing.py     # Image preprocessing
│   ├── detection.py         # Text detection
│   ├── recognition.py       # Handwriting recognition
│   ├── postprocess.py       # Price/currency extraction
│   ├── optimization.py      # Pairing algorithms
│   └── config.py            # Configuration management
├── training/                 # Model training infrastructure
│   ├── train_synthetic.py   # Training script for synthetic data
│   ├── train_trocr.py       # Training script for real data
│   ├── evaluate_model.py    # Model evaluation
│   ├── extract_crops.py     # Extract training samples
│   ├── label_tool.py        # Interactive labeling
│   └── requirements_training.txt
├── dataset_generator/        # Synthetic dataset creation
│   └── synthetic/
│       ├── generate_synthetic_dataset.py
│       ├── vocabulary.py    # Menu item vocabulary
│       ├── augment.py       # Image augmentation
│       └── dataset_synth/   # Generated dataset (4,393 samples)
├── data/samples/            # Test menu photos (gitignored)
├── docs/                    # Comprehensive documentation
├── tests/                   # Test suite
└── Experiment/              # Pairing algorithm experiments
```

## Documentation

Detailed documentation is available in the `docs/` directory:

- **[handwritten-menu-scanner-spec.md](docs/handwritten-menu-scanner-spec.md)** — Complete project specification including architecture decisions, constraints, accuracy targets, and rollout plan
- **[recognition-status-report.md](docs/recognition-status-report.md)** — Current implementation status with real-world test results and known issues
- **[code-analysis-report.md](docs/code-analysis-report.md)** — Comprehensive code review covering strengths, blind spots, and improvement recommendations
- **[analysis-summary.md](docs/analysis-summary.md)** — Executive summary of code quality and production readiness
- **[action-items.md](docs/action-items.md)** — Prioritized list of improvements with code examples

## Key Features

- **Real-world validation:** Tested on actual handwritten menus with promising results
- **Custom model training:** Fine-tune TrOCR on your own menu data
- **Synthetic dataset:** Pre-generated 4,393 handwritten menu samples for training
- **Flexible currency handling:** Supports TND and EUR with menu-level defaults
- **Confidence scoring:** Highlights low-confidence fields for focused review
- **Position-based pairing:** Robust item-to-price matching even with OCR errors
- **GPU acceleration:** Optimized for NVIDIA GPUs with CUDA support
- **Windows-optimized:** Training pipeline configured for Windows laptops

## Technology Stack

- **Deep Learning:** PyTorch 2.0+, Transformers (TrOCR), Hugging Face Accelerate
- **Computer Vision:** OpenCV, PaddleOCR
- **Synthetic Data:** Handwriting synthesis with neural networks
- **Language:** Python 3.10+
- **Hardware:** CPU or NVIDIA GPU (4GB+ VRAM for training)

## Requirements

### Pipeline Usage
- Python 3.10 or higher
- 4GB RAM minimum, 8GB+ recommended
- Windows/Linux/macOS

### Model Training
- Python 3.10+
- NVIDIA GPU with 4GB+ VRAM (RTX 3050 or better)
- CUDA 11.8+
- 16GB+ RAM recommended
- 10GB+ free disk space for checkpoints

See [requirements.txt](requirements.txt) for pipeline dependencies and [training/requirements_training.txt](training/requirements_training.txt) for training dependencies.

## Development

### Pipeline Development

Install development dependencies for testing and code quality tools:

```bash
pip install -r requirements-dev.txt
```

### Training New Models

1. **Generate synthetic dataset** (if needed):
```bash
cd dataset_generator/synthetic
python generate_synthetic_dataset.py \
    --out_dir dataset_synth \
    --repo_dir ../handwriting-synthesis-master \
    --samples_per_field 10
```

2. **Train on synthetic data**:
```bash
cd training
python train_synthetic.py \
    --dataset_dir ../dataset_generator/synthetic/dataset_synth \
    --output_model_dir ../models/trocr_menu_v1 \
    --batch_size 4 \
    --max_epochs 5 \
    --eval_test
```

3. **Collect and label real data**:
```bash
python extract_crops.py --images_dir ../data/real_menus --out_dir ../data/real_crops
python label_tool.py --out_dir ../data/real_crops
```

4. **Fine-tune on mixed data**:
```bash
python train_trocr.py \
    --out_dir ../data/mixed_dataset \
    --output_model_dir ../models/trocr_menu_v2
```

Current development priorities:
1. ✅ Synthetic dataset generation
2. ✅ TrOCR fine-tuning infrastructure
3. ✅ Windows-optimized training pipeline
4. 🔄 Real data collection and labeling
5. 🔄 Mixed synthetic + real training
6. 📋 Comprehensive test suite
7. 📋 Production hardening (logging, monitoring)

See [training/README.md](training/README.md) for detailed training documentation.

## Design Philosophy

This project embraces realistic accuracy expectations over perfection. The goal is to provide a useful draft that reduces manual typing effort, not to achieve 100% automated extraction. The human review step is a first-class feature, not a fallback.

Key principles:
- Build for correction, not perfection
- Validate with real-world data, not synthetic tests
- Document limitations honestly
- Learn from failures and iterate

## Contributing

Contributions are welcome. Please review the code analysis reports in `docs/` to understand current priorities and code quality standards.



---

**Made with dedication by Hamza Z**
