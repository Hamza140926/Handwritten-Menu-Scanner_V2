"""
Custom exception hierarchy for the handwritten menu scanner pipeline.

Provides specific exceptions for each pipeline stage to enable:
- Targeted error handling
- Clear error messages
- Graceful degradation
- Better debugging

Usage:
    from exceptions import PreprocessingError, DetectionError
    
    try:
        result = preprocess_image(path)
    except PreprocessingError as e:
        logger.error("Preprocessing failed", exc_info=True)
        # Handle or return partial result
"""


class PipelineError(Exception):
    """Base exception for all pipeline errors."""
    pass


class PreprocessingError(PipelineError):
    """
    Raised when image preprocessing fails.
    
    Common causes:
    - Cannot load image (corrupt file)
    - Deskewing fails
    - Contrast normalization fails
    """
    pass


class DetectionError(PipelineError):
    """
    Raised when text region detection fails.
    
    Common causes:
    - PaddleOCR model fails to load
    - Detection returns no regions
    - Detector crashes on specific image
    """
    pass


class RecognitionError(PipelineError):
    """
    Raised when handwriting recognition fails.
    
    Common causes:
    - TrOCR model fails to load
    - Out of memory (GPU/CPU)
    - Recognition crashes on specific crop
    """
    pass


class PostprocessingError(PipelineError):
    """
    Raised when postprocessing fails.
    
    Common causes:
    - Price extraction regex fails
    - Currency resolution fails
    """
    pass


class AssemblyError(PipelineError):
    """
    Raised when menu assembly (item pairing) fails.
    
    Common causes:
    - Column splitting fails
    - Item pairing logic fails
    - Invalid data structure
    """
    pass
