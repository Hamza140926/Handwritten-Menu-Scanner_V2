"""
Tests for exceptions.py - custom exception hierarchy.
"""

import pytest
from exceptions import (
    PipelineError,
    PreprocessingError,
    DetectionError,
    RecognitionError,
    PostprocessingError,
    AssemblyError,
)


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""
    
    def test_all_inherit_from_pipeline_error(self):
        """Test all custom exceptions inherit from PipelineError."""
        assert issubclass(PreprocessingError, PipelineError)
        assert issubclass(DetectionError, PipelineError)
        assert issubclass(RecognitionError, PipelineError)
        assert issubclass(PostprocessingError, PipelineError)
        assert issubclass(AssemblyError, PipelineError)
    
    def test_pipeline_error_is_exception(self):
        """Test PipelineError inherits from Exception."""
        assert issubclass(PipelineError, Exception)
    
    def test_can_catch_specific_exceptions(self):
        """Test specific exceptions can be caught independently."""
        with pytest.raises(PreprocessingError):
            raise PreprocessingError("test")
        
        with pytest.raises(DetectionError):
            raise DetectionError("test")
        
        with pytest.raises(RecognitionError):
            raise RecognitionError("test")
    
    def test_can_catch_all_with_pipeline_error(self):
        """Test all pipeline exceptions can be caught with base class."""
        with pytest.raises(PipelineError):
            raise PreprocessingError("test")
        
        with pytest.raises(PipelineError):
            raise DetectionError("test")
        
        with pytest.raises(PipelineError):
            raise RecognitionError("test")


class TestExceptionMessages:
    """Tests for exception message handling."""
    
    def test_preprocessing_error_message(self):
        """Test PreprocessingError preserves message."""
        msg = "Failed to load image"
        with pytest.raises(PreprocessingError, match=msg):
            raise PreprocessingError(msg)
    
    def test_detection_error_message(self):
        """Test DetectionError preserves message."""
        msg = "No regions detected"
        with pytest.raises(DetectionError, match=msg):
            raise DetectionError(msg)
    
    def test_recognition_error_message(self):
        """Test RecognitionError preserves message."""
        msg = "Model failed to load"
        with pytest.raises(RecognitionError, match=msg):
            raise RecognitionError(msg)
    
    def test_postprocessing_error_message(self):
        """Test PostprocessingError preserves message."""
        msg = "Price extraction failed"
        with pytest.raises(PostprocessingError, match=msg):
            raise PostprocessingError(msg)
    
    def test_assembly_error_message(self):
        """Test AssemblyError preserves message."""
        msg = "Column splitting failed"
        with pytest.raises(AssemblyError, match=msg):
            raise AssemblyError(msg)


class TestExceptionChaining:
    """Tests for exception chaining (from clause)."""
    
    def test_can_chain_exceptions(self):
        """Test exceptions can be chained with 'from' clause."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise PreprocessingError("Preprocessing failed") from e
        except PreprocessingError as pe:
            assert pe.__cause__ is not None
            assert isinstance(pe.__cause__, ValueError)
            assert str(pe.__cause__) == "Original error"
    
    def test_chained_exception_traceback(self):
        """Test chained exceptions preserve original traceback."""
        with pytest.raises(PreprocessingError) as exc_info:
            try:
                raise IOError("File not found")
            except IOError as e:
                raise PreprocessingError("Cannot load image") from e
        
        # Check that original exception is accessible
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, IOError)


class TestErrorHandlingPatterns:
    """Tests for common error handling patterns."""
    
    def test_catch_specific_then_reraise_as_pipeline_error(self):
        """Test pattern: catch specific error, reraise as PipelineError."""
        def risky_operation():
            raise FileNotFoundError("File missing")
        
        with pytest.raises(PreprocessingError):
            try:
                risky_operation()
            except FileNotFoundError as e:
                raise PreprocessingError(f"Operation failed: {e}") from e
    
    def test_multiple_exception_types_in_one_handler(self):
        """Test catching multiple pipeline exception types."""
        def may_fail(error_type):
            if error_type == "preprocessing":
                raise PreprocessingError("Preprocessing failed")
            elif error_type == "detection":
                raise DetectionError("Detection failed")
            else:
                raise RecognitionError("Recognition failed")
        
        for error_type in ["preprocessing", "detection", "recognition"]:
            with pytest.raises(PipelineError):
                may_fail(error_type)
    
    def test_exception_message_formatting(self):
        """Test that exception messages can include context."""
        path = "/path/to/image.jpg"
        error_msg = f"Failed to process {path}"
        
        with pytest.raises(PreprocessingError, match=error_msg):
            raise PreprocessingError(error_msg)


class TestExceptionDocstrings:
    """Tests that exceptions have proper documentation."""
    
    def test_all_exceptions_have_docstrings(self):
        """Test all exception classes have docstrings."""
        exceptions = [
            PipelineError,
            PreprocessingError,
            DetectionError,
            RecognitionError,
            PostprocessingError,
            AssemblyError,
        ]
        
        for exc_class in exceptions:
            assert exc_class.__doc__ is not None
            assert len(exc_class.__doc__.strip()) > 0


class TestExceptionInstantiation:
    """Tests for creating exception instances."""
    
    def test_can_create_without_message(self):
        """Test exceptions can be created without message."""
        exc = PreprocessingError()
        assert isinstance(exc, PreprocessingError)
    
    def test_can_create_with_empty_message(self):
        """Test exceptions can be created with empty message."""
        exc = DetectionError("")
        assert isinstance(exc, DetectionError)
    
    def test_can_create_with_long_message(self):
        """Test exceptions can handle long messages."""
        long_msg = "x" * 1000
        exc = RecognitionError(long_msg)
        assert str(exc) == long_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
