"""
Tests for postprocess.py - price extraction and currency resolution.

These are the most critical tests since price extraction is core functionality.
"""

import pytest
from postprocess import (
    extract_price,
    detect_currency_symbol,
    resolve_currency,
    process_region,
    SUPPORTED_CURRENCIES,
    DEFAULT_CURRENCY,
)


class TestPriceExtraction:
    """Tests for extract_price function."""
    
    def test_decimal_dot(self):
        """Test price with dot as decimal separator."""
        result = extract_price("12.50")
        assert result["value"] == 12.5
        assert result["raw"] == "12.50"
        assert not result["ambiguous"]
    
    def test_decimal_comma(self):
        """Test price with comma as decimal separator (European style)."""
        result = extract_price("12,50")
        assert result["value"] == 12.5
        assert result["raw"] == "12,50"
        assert not result["ambiguous"]
    
    def test_tunisian_millime_format(self):
        """Test Tunisian 3-decimal format (e.g., 12.500)."""
        result = extract_price("12.500")
        assert result["value"] == 12.5
        assert result["raw"] == "12.500"
        assert not result["ambiguous"]
    
    def test_integer_only(self):
        """Test whole number without decimals."""
        result = extract_price("15")
        assert result["value"] == 15.0
        assert result["raw"] == "15"
        assert not result["ambiguous"]
    
    def test_price_in_text(self):
        """Test extracting price from text with item name."""
        result = extract_price("Espresso 3.0")
        assert result["value"] == 3.0
        assert result["raw"] == "3.0"
        assert not result["ambiguous"]
    
    def test_no_price(self):
        """Test text with no numeric price."""
        result = extract_price("Cappuccino")
        assert result["value"] is None
        assert result["raw"] is None
        assert not result["ambiguous"]
    
    def test_multiple_numbers_ambiguous(self):
        """Test string with multiple different numbers (noisy OCR)."""
        result = extract_price("3, 3, 5.")
        assert result["ambiguous"] is True
        assert result["value"] is not None  # Should still extract one
    
    def test_repeated_numbers_not_ambiguous(self):
        """Test repeated same number (OCR duplication, not ambiguous choice)."""
        result = extract_price("3, 3, 3.")
        # All same value, so not truly ambiguous
        assert result["value"] == 3.0
    
    def test_prefer_decimal_over_integer(self):
        """Test that decimal numbers are preferred over integers."""
        result = extract_price("3 12.500")
        assert result["value"] == 12.5  # Should pick 12.500, not 3
        assert "." in result["raw"] or "," in result["raw"]
    
    def test_zero_price(self):
        """Test edge case of zero price."""
        result = extract_price("0.00")
        assert result["value"] == 0.0
    
    def test_large_price(self):
        """Test large price value."""
        result = extract_price("999.99")
        assert result["value"] == 999.99
    
    def test_with_currency_symbols(self):
        """Test price extraction ignores currency letters."""
        result = extract_price("12.500 DT")
        assert result["value"] == 12.5
        
        result = extract_price("€5.00")
        assert result["value"] == 5.0


class TestCurrencyDetection:
    """Tests for detect_currency_symbol function."""
    
    def test_detect_euro_symbol(self):
        """Test detection of € symbol."""
        assert detect_currency_symbol("€5.00") == "EUR"
        assert detect_currency_symbol("5.00€") == "EUR"
    
    def test_no_currency_symbol(self):
        """Test when no currency symbol present."""
        assert detect_currency_symbol("12.50") is None
        assert detect_currency_symbol("12.50 DT") is None  # Letters ignored
    
    def test_mixed_text(self):
        """Test currency detection in mixed text."""
        assert detect_currency_symbol("Coffee €3.50") == "EUR"


class TestCurrencyResolution:
    """Tests for resolve_currency function."""
    
    def test_default_currency_no_symbol(self):
        """Test default currency used when no symbol detected."""
        currency, source = resolve_currency("TND", "12.50", confidence=0.9)
        assert currency == "TND"
        assert source == "default"
    
    def test_symbol_overrides_with_high_confidence(self):
        """Test currency symbol overrides default when confidence is high."""
        currency, source = resolve_currency("TND", "€5.00", confidence=0.9)
        assert currency == "EUR"
        assert source == "detected_symbol"
    
    def test_symbol_ignored_with_low_confidence(self):
        """Test currency symbol ignored when confidence is low."""
        currency, source = resolve_currency("TND", "€5.00", confidence=0.5)
        assert currency == "TND"
        assert source == "default"
    
    def test_threshold_boundary(self):
        """Test currency resolution at confidence threshold."""
        # At exactly the threshold, should use symbol
        currency, source = resolve_currency("TND", "€5.00", confidence=0.85)
        assert currency == "EUR"
        
        # Just below threshold, should use default
        currency, source = resolve_currency("TND", "€5.00", confidence=0.84)
        assert currency == "TND"


class TestProcessRegion:
    """Tests for process_region function."""
    
    def test_complete_region_processing(self):
        """Test full processing of a region with all fields."""
        region = {
            "text": "Espresso 3.50",
            "confidence": 0.92,
            "box": [[0, 0], [100, 0], [100, 50], [0, 50]],
            "y": 100,
            "x": 50,
        }
        
        result = process_region(region, default_currency="TND")
        
        assert result["text"] == "Espresso 3.50"
        assert result["confidence"] == 0.92
        assert result["price_value"] == 3.5
        assert result["price_raw"] == "3.50"
        assert not result["price_ambiguous"]
        assert result["currency"] == "TND"
        assert result["currency_source"] == "default"
        # Original fields preserved
        assert result["box"] is not None
        assert result["y"] == 100
        assert result["x"] == 50
    
    def test_region_with_no_price(self):
        """Test region containing only text, no price."""
        region = {"text": "Coffee", "confidence": 0.95}
        result = process_region(region)
        
        assert result["price_value"] is None
        assert result["price_raw"] is None
        assert not result["price_ambiguous"]
    
    def test_region_with_euro_symbol(self):
        """Test region with € symbol and high confidence."""
        region = {"text": "€5.00", "confidence": 0.92}
        result = process_region(region, default_currency="TND")
        
        assert result["currency"] == "EUR"
        assert result["currency_source"] == "detected_symbol"


class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_empty_string(self):
        """Test empty string input."""
        result = extract_price("")
        assert result["value"] is None
    
    def test_special_characters(self):
        """Test string with special characters."""
        result = extract_price("###12.50###")
        assert result["value"] == 12.5
    
    def test_very_long_decimal(self):
        """Test price with too many decimal places."""
        result = extract_price("12.12345")
        # Should still extract something
        assert result["value"] is not None
    
    def test_negative_price(self):
        """Test negative number (shouldn't match in real menus)."""
        # Our regex doesn't match negative numbers, which is correct
        result = extract_price("-5.00")
        assert result["value"] == 5.0  # Matches 5.00, ignores -


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
