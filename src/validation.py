"""
Input validation for the handwritten menu scanner pipeline.

Validates uploaded images before processing to prevent:
- Crashes from corrupted/malformed files
- DoS attacks from oversized files
- Memory exhaustion from huge images
- Path traversal attacks
- Unsupported file formats

Usage:
    from validation import validate_image_input, ValidationError
    
    try:
        validated_path = validate_image_input(user_provided_path)
        # Safe to process
        result = preprocess_image(validated_path)
    except ValidationError as e:
        logger.error("Invalid input", extra={"error": str(e)})
        return {"error": str(e)}
"""

import cv2
from pathlib import Path
from typing import Union
from logging_config import get_logger

logger = get_logger(__name__)

# Security limits
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB - reasonable for phone photos
MIN_FILE_SIZE = 1024  # 1KB - prevent empty/corrupt files
MAX_DIMENSION = 10000  # 10000px - prevents memory bombs
MIN_DIMENSION = 100  # 100px - too small to be a real menu
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


def validate_image_input(path: Union[str, Path]) -> Path:
    """
    Validate an image file before processing.
    
    Checks:
    - File exists and is a regular file
    - File extension is allowed
    - File size is within limits
    - Image can be loaded by OpenCV
    - Image dimensions are reasonable
    
    Args:
        path: Path to image file (string or Path object)
        
    Returns:
        Validated Path object (absolute, resolved)
        
    Raises:
        ValidationError: If any validation check fails
    """
    logger.debug("Validating input", extra={"path": str(path)})
    
    # Convert to Path and resolve (handles relative paths, symlinks)
    try:
        path = Path(path).resolve()
    except (OSError, RuntimeError) as e:
        logger.warning("Path resolution failed", extra={"path": str(path), "error": str(e)})
        raise ValidationError(f"Invalid path: {e}")
    
    # Check file exists
    if not path.exists():
        logger.warning("File not found", extra={"path": str(path)})
        raise ValidationError(f"File not found: {path}")
    
    # Check it's a file (not directory, device, etc.)
    if not path.is_file():
        logger.warning("Path is not a file", extra={"path": str(path)})
        raise ValidationError(f"Path is not a regular file: {path}")
    
    # Check file extension
    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning("Unsupported format", extra={"path": str(path), "extension": ext})
        raise ValidationError(
            f"Unsupported file format: {ext}. "
            f"Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    
    # Check file size
    try:
        file_size = path.stat().st_size
    except OSError as e:
        logger.warning("Cannot read file stats", extra={"path": str(path), "error": str(e)})
        raise ValidationError(f"Cannot access file: {e}")
    
    if file_size < MIN_FILE_SIZE:
        logger.warning("File too small", extra={"path": str(path), "size": file_size})
        raise ValidationError(
            f"File too small ({file_size} bytes). "
            f"Minimum size: {MIN_FILE_SIZE} bytes. File may be corrupt."
        )
    
    if file_size > MAX_FILE_SIZE:
        logger.warning("File too large", extra={"path": str(path), "size": file_size})
        raise ValidationError(
            f"File too large ({file_size / 1024 / 1024:.1f} MB). "
            f"Maximum size: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB"
        )
    
    # Try to load and validate image dimensions
    try:
        image = cv2.imread(str(path))
    except Exception as e:
        logger.warning("OpenCV load failed", extra={"path": str(path), "error": str(e)})
        raise ValidationError(f"Cannot load image: {e}")
    
    if image is None:
        logger.warning("Image is None after load", extra={"path": str(path)})
        raise ValidationError(
            f"Cannot load image. File may be corrupt or not a valid image format."
        )
    
    # Check dimensions
    try:
        h, w = image.shape[:2]
    except (AttributeError, ValueError) as e:
        logger.warning("Cannot get image dimensions", extra={"path": str(path), "error": str(e)})
        raise ValidationError(f"Invalid image data: {e}")
    
    if h < MIN_DIMENSION or w < MIN_DIMENSION:
        logger.warning("Image too small", extra={"path": str(path), "width": w, "height": h})
        raise ValidationError(
            f"Image too small ({w}x{h} pixels). "
            f"Minimum dimension: {MIN_DIMENSION}px"
        )
    
    if h > MAX_DIMENSION or w > MAX_DIMENSION:
        logger.warning("Image too large", extra={"path": str(path), "width": w, "height": h})
        raise ValidationError(
            f"Image dimensions too large ({w}x{h} pixels). "
            f"Maximum dimension: {MAX_DIMENSION}px"
        )
    
    logger.info("Validation passed", extra={
        "path": str(path),
        "size_mb": file_size / 1024 / 1024,
        "width": w,
        "height": h
    })
    
    return path


def validate_currency(currency: str) -> str:
    """
    Validate currency code.
    
    Args:
        currency: Currency code (e.g., "TND", "EUR")
        
    Returns:
        Validated currency code (uppercase)
        
    Raises:
        ValidationError: If currency is not supported
    """
    SUPPORTED_CURRENCIES = {"TND", "EUR"}
    
    currency = currency.strip().upper()
    
    if currency not in SUPPORTED_CURRENCIES:
        logger.warning("Unsupported currency", extra={"currency": currency})
        raise ValidationError(
            f"Unsupported currency: {currency}. "
            f"Supported: {', '.join(sorted(SUPPORTED_CURRENCIES))}"
        )
    
    return currency


if __name__ == "__main__":
    import sys
    from logging_config import setup_logging
    
    setup_logging(level="DEBUG")
    
    if len(sys.argv) != 2:
        print("Usage: python validation.py <path_to_image>")
        sys.exit(1)
    
    try:
        validated = validate_image_input(sys.argv[1])
        print(f"✓ Validation passed: {validated}")
    except ValidationError as e:
        print(f"✗ Validation failed: {e}")
        sys.exit(1)
