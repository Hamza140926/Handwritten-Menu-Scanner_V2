# Installation Guide

## Prerequisites
- Python 3.10+
- CUDA 13.0 compatible GPU (optional, for faster processing)
- Anaconda or Miniconda

## Installation

### 1. Clone/Download the Project
```bash
cd C:\Users\zussl\Desktop\scantosee\scantosee_APP\handwritten-menu-scanner
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
conda create -n menu-scanner python=3.10
conda activate menu-scanner
```

### 3. Install Dependencies

**For GPU (with CUDA 13.0):**
```bash
pip install -r requirements.txt
```

**For CPU only:**
```bash
# Uninstall GPU torch if present
pip uninstall torch

# Install CPU version
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "from transformers import TrOCRProcessor; print('Transformers OK')"
python -c "from paddleocr import PaddleOCR; print('PaddleOCR OK')"
```

## For Development

Install development dependencies (testing, linting, formatting):
```bash
pip install -r requirements-dev.txt
```

## Troubleshooting

### "No module named 'sentencepiece'"
```bash
pip install sentencepiece==0.2.2
```

### CUDA not available
- Check NVIDIA driver: `nvidia-smi`
- Reinstall torch with correct CUDA version
- Or use CPU version (slower but works)

### PaddleOCR errors
```bash
pip install paddlepaddle==3.3.1 paddleocr==3.7.0 --force-reinstall
```
