# Handwritten Menu Scanner

An AI-powered pipeline that converts handwritten paper menus into structured, editable digital formats. Built for the Scan2See application, this system helps restaurant and cafe owners digitize their menus without manual data entry.

## Overview

The scanner processes photos of handwritten menus through a multi-stage pipeline: image preprocessing, text detection, handwriting recognition, and intelligent item-price pairing. The system is designed for real-world constraints—general handwriting styles, messy layouts, and varying lighting conditions—achieving practical accuracy suitable for human review and correction.

**Current Status:** Working baseline with all core components implemented and validated against real menu photographs.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/hamza030220/Handwritten-Menu-Scanner_V2.git
cd Handwritten-Menu-Scanner_V2

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python src/pipeline.py path/to/menu_photo.jpg
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
├── src/               # Core pipeline implementation
├── data/samples/      # Test menu photos (gitignored)
├── notebooks/         # Experimental notebooks
├── docs/              # Comprehensive documentation
└── tests/             # Test suite (in development)
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
- **Flexible currency handling:** Supports TND and EUR with menu-level defaults
- **Confidence scoring:** Highlights low-confidence fields for focused review
- **Position-based pairing:** Robust item-to-price matching even with OCR errors
- **GPU acceleration:** Optional CUDA support for faster processing

## Technology Stack

- **Deep Learning:** PyTorch, Transformers (TrOCR)
- **Computer Vision:** OpenCV, PaddleOCR
- **Language:** Python 3.10+
- **Hardware:** CPU or NVIDIA GPU (4GB+ VRAM recommended)

## Requirements

- Python 3.10 or higher
- CUDA 13.0+ (optional, for GPU acceleration)
- 4GB RAM minimum, 8GB+ recommended
- Windows/Linux/macOS

See [requirements.txt](requirements.txt) for complete dependency list.

## Development

Install development dependencies for testing and code quality tools:

```bash
pip install -r requirements-dev.txt
```

Current development priorities:
1. Comprehensive test suite implementation
2. Error handling and logging infrastructure
3. Performance benchmarking and optimization
4. Production hardening (security, validation, monitoring)

See [action-items.md](docs/action-items.md) for detailed development roadmap.

## Design Philosophy

This project embraces realistic accuracy expectations over perfection. The goal is to provide a useful draft that reduces manual typing effort, not to achieve 100% automated extraction. The human review step is a first-class feature, not a fallback.

Key principles:
- Build for correction, not perfection
- Validate with real-world data, not synthetic tests
- Document limitations honestly
- Learn from failures and iterate

## Contributing

Contributions are welcome. Please review the code analysis reports in `docs/` to understand current priorities and code quality standards.

## License

[Add your license here]

---

**Made with dedication by Hamza Z**
