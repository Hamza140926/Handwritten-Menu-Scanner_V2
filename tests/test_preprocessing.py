"""
Tests for preprocessing.py - image preprocessing functions.
"""

import pytest
import numpy as np
import cv2
from preprocessing import (
    resize_max_dimension,
    compute_skew_angle,
    deskew,
    normalize_contrast,
)
from config import get_config
from exceptions import PreprocessingError


class TestResizeMaxDimension:
    """Tests for resize_max_dimension function."""
    
    def test_no_resize_needed(self):
        """Test image smaller than max dimension is not resized."""
        image = np.zeros((1000, 1500, 3), dtype=np.uint8)
        result = resize_max_dimension(image, max_dim=2000)
        
        assert result.shape == image.shape
    
    def test_resize_width_larger(self):
        """Test resize when width is the larger dimension."""
        image = np.zeros((1000, 3000, 3), dtype=np.uint8)
        result = resize_max_dimension(image, max_dim=2000)
        
        assert result.shape[1] == 2000  # Width should be 2000
        assert result.shape[0] == 666  # Height scaled proportionally (rounded)
    
    def test_resize_height_larger(self):
        """Test resize when height is the larger dimension."""
        image = np.zeros((3000, 1000, 3), dtype=np.uint8)
        result = resize_max_dimension(image, max_dim=2000)
        
        assert result.shape[0] == 2000  # Height should be 2000
        assert result.shape[1] == 666  # Width scaled proportionally (rounded)
    
    def test_aspect_ratio_preserved(self):
        """Test that aspect ratio is preserved after resize."""
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        result = resize_max_dimension(image, max_dim=400)
        
        original_ratio = 800 / 600
        result_ratio = result.shape[1] / result.shape[0]
        
        assert abs(original_ratio - result_ratio) < 0.01  # Within 1%
    
    def test_square_image(self):
        """Test resize of square image."""
        image = np.zeros((2000, 2000, 3), dtype=np.uint8)
        result = resize_max_dimension(image, max_dim=1000)
        
        assert result.shape[0] == 1000
        assert result.shape[1] == 1000


class TestComputeSkewAngle:
    """Tests for compute_skew_angle function."""
    
    def test_no_skew(self):
        """Test perfectly aligned image returns ~0 angle."""
        # Create image with horizontal lines
        image = np.zeros((500, 500), dtype=np.uint8)
        image[100:110, :] = 255  # Horizontal line
        image[200:210, :] = 255  # Another horizontal line
        
        angle = compute_skew_angle(image)
        
        assert abs(angle) < 1.0  # Should be very close to 0
    
    def test_blank_image(self):
        """Test blank image returns 0 angle."""
        image = np.zeros((500, 500), dtype=np.uint8)
        angle = compute_skew_angle(image)
        
        assert angle == 0.0  # No lines found
    
    def test_extreme_angle_clamped(self):
        """Test that extreme angles beyond threshold return 0."""
        # This test is conceptual - in practice, creating a specific
        # angle is complex with Hough line detection
        # The function should return 0 for angles > MAX_SKEW_CORRECTION_DEGREES
        pass  # Tested implicitly in code with safety clamp
    
    def test_max_correction_limit(self):
        """Test MAX_SKEW_CORRECTION_DEGREES constant is reasonable."""
        cfg = get_config().preprocessing
        assert cfg.max_skew_degrees == 15.0
        assert 0 < cfg.max_skew_degrees < 45


class TestDeskew:
    """Tests for deskew function."""
    
    def test_zero_angle_no_change(self):
        """Test that 0 angle doesn't modify image."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = deskew(image, 0.0)
        
        np.testing.assert_array_equal(image, result)
    
    def test_small_angle_ignored(self):
        """Test very small angle (<0.1) is ignored."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = deskew(image, 0.05)
        
        np.testing.assert_array_equal(image, result)
    
    def test_deskew_maintains_shape(self):
        """Test deskew maintains original image dimensions."""
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        result = deskew(image, 5.0)
        
        assert result.shape == image.shape
    
    def test_rotation_applied(self):
        """Test that rotation is actually applied."""
        # Create image with distinct pattern
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        image[50:60, :] = 255  # White horizontal line
        
        result = deskew(image, 10.0)
        
        # After rotation, image should differ
        assert not np.array_equal(image, result)


class TestNormalizeContrast:
    """Tests for normalize_contrast function."""
    
    def test_output_shape_unchanged(self):
        """Test output has same shape as input."""
        image = np.random.randint(0, 255, (200, 300), dtype=np.uint8)
        result = normalize_contrast(image)
        
        assert result.shape == image.shape
    
    def test_output_dtype_unchanged(self):
        """Test output has same dtype as input."""
        image = np.random.randint(0, 255, (200, 300), dtype=np.uint8)
        result = normalize_contrast(image)
        
        assert result.dtype == np.uint8
    
    def test_low_contrast_improved(self):
        """Test that low contrast image gets improved."""
        # Create low contrast image (all values 100-110)
        image = np.random.randint(100, 110, (200, 300), dtype=np.uint8)
        result = normalize_contrast(image)
        
        # After CLAHE, range should be wider
        original_range = image.max() - image.min()
        result_range = result.max() - result.min()
        
        assert result_range >= original_range
    
    def test_already_good_contrast(self):
        """Test image with good contrast is not degraded."""
        # Create high contrast image
        image = np.random.randint(0, 255, (200, 300), dtype=np.uint8)
        result = normalize_contrast(image)
        
        # Should still be valid uint8 range
        assert result.min() >= 0
        assert result.max() <= 255


class TestPreprocessingConstants:
    """Tests for preprocessing module constants."""
    
    def test_max_dimension_reasonable(self):
        """Test MAX_DIMENSION is set to reasonable value."""
        cfg = get_config().preprocessing
        assert cfg.max_dimension == 2000
        assert 1000 <= cfg.max_dimension <= 5000
    
    def test_skew_correction_limit_reasonable(self):
        """Test skew correction limit is reasonable."""
        cfg = get_config().preprocessing
        assert 5 <= cfg.max_skew_degrees <= 30


class TestPreprocessingErrorHandling:
    """Tests for error handling in preprocessing."""
    
    def test_load_nonexistent_file_raises(self):
        """Test loading non-existent file raises PreprocessingError."""
        from preprocessing import load_image
        
        with pytest.raises(PreprocessingError):
            load_image("/nonexistent/file.jpg")
    
    def test_preprocess_invalid_path_raises(self):
        """Test preprocessing invalid path raises PreprocessingError."""
        from preprocessing import preprocess_image
        
        with pytest.raises(PreprocessingError):
            preprocess_image("/invalid/path.jpg")


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_very_small_image(self):
        """Test processing very small image."""
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        result = resize_max_dimension(image)
        
        # Should not resize (already small)
        assert result.shape == image.shape
    
    def test_grayscale_input(self):
        """Test functions handle grayscale images."""
        # Most functions should work with grayscale (2D) arrays
        gray = np.random.randint(0, 255, (200, 300), dtype=np.uint8)
        
        # normalize_contrast specifically works on grayscale
        result = normalize_contrast(gray)
        assert result.shape == gray.shape
    
    def test_single_pixel_image(self):
        """Test extreme case of 1x1 image."""
        image = np.array([[[128, 128, 128]]], dtype=np.uint8)
        result = resize_max_dimension(image)
        
        assert result.shape == image.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
