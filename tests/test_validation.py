"""
Tests for validation.py - input validation and security checks.
"""

import pytest
import tempfile
from pathlib import Path
from validation import (
    validate_image_input,
    validate_currency,
    ValidationError,
)
from config import get_config


class TestValidateImageInput:
    """Tests for validate_image_input function."""
    
    def test_nonexistent_file(self):
        """Test validation fails for non-existent file."""
        with pytest.raises(ValidationError, match="File not found"):
            validate_image_input("/nonexistent/path/image.jpg")
    
    def test_unsupported_extension(self, tmp_path):
        """Test validation fails for unsupported file extension."""
        # Create a .txt file
        test_file = tmp_path / "test.txt"
        test_file.write_text("not an image")
        
        with pytest.raises(ValidationError, match="Unsupported file format"):
            validate_image_input(str(test_file))
    
    def test_file_too_small(self, tmp_path):
        """Test validation fails for files below minimum size."""
        # Create a tiny file
        test_file = tmp_path / "tiny.jpg"
        test_file.write_bytes(b"x" * 100)  # 100 bytes, below 1KB minimum
        
        with pytest.raises(ValidationError, match="File too small"):
            validate_image_input(str(test_file))
    
    def test_supported_extensions(self):
        """Test that common image extensions are supported."""
        cfg = get_config().validation
        supported = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        # This test just verifies the extensions are recognized
        for ext in supported:
            assert ext in cfg.allowed_extensions


class TestValidateCurrency:
    """Tests for validate_currency function."""
    
    def test_valid_tnd(self):
        """Test TND currency validation."""
        result = validate_currency("TND")
        assert result == "TND"
        
        # Test lowercase
        result = validate_currency("tnd")
        assert result == "TND"
    
    def test_valid_eur(self):
        """Test EUR currency validation."""
        result = validate_currency("EUR")
        assert result == "EUR"
        
        # Test lowercase
        result = validate_currency("eur")
        assert result == "EUR"
    
    def test_invalid_currency(self):
        """Test validation fails for unsupported currency."""
        with pytest.raises(ValidationError, match="Unsupported currency"):
            validate_currency("USD")
        
        with pytest.raises(ValidationError, match="Unsupported currency"):
            validate_currency("FAKE")
    
    def test_currency_with_whitespace(self):
        """Test currency validation strips whitespace."""
        result = validate_currency("  TND  ")
        assert result == "TND"
    
    def test_empty_currency(self):
        """Test validation fails for empty currency."""
        with pytest.raises(ValidationError):
            validate_currency("")


class TestSecurityLimits:
    """Tests for security limit constants."""
    
    def test_file_size_limits_reasonable(self):
        """Test file size limits are reasonable."""
        cfg = get_config().validation
        assert cfg.min_file_size == 1024  # 1KB
        assert cfg.max_file_size == 50 * 1024 * 1024  # 50MB
        assert cfg.min_file_size < cfg.max_file_size
    
    def test_dimension_limits_reasonable(self):
        """Test dimension limits are reasonable."""
        cfg = get_config().validation
        assert cfg.min_dimension == 100  # 100px
        assert cfg.max_dimension == 10000  # 10000px
        assert cfg.min_dimension < cfg.max_dimension


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
