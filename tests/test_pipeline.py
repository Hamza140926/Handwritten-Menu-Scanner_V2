"""
Tests for pipeline.py - menu assembly and item pairing logic.
"""

import pytest
from pipeline import (
    split_columns,
    identify_name_and_price_columns,
    pair_items,
    _price_hit_rate,
)


class TestSplitColumns:
    """Tests for split_columns function."""
    
    def test_clear_two_column_layout(self):
        """Test splitting clear two-column layout."""
        regions = [
            {"x": 100, "y": 100},  # Left column
            {"x": 105, "y": 150},  # Left column
            {"x": 300, "y": 100},  # Right column (large gap)
            {"x": 305, "y": 150},  # Right column
        ]
        
        left, right = split_columns(regions)
        
        assert len(left) == 2
        assert len(right) == 2
        assert all(r["x"] < 200 for r in left)
        assert all(r["x"] > 200 for r in right)
    
    def test_single_column_no_split(self):
        """Test single column returns no split."""
        regions = [
            {"x": 100, "y": 100},
            {"x": 110, "y": 150},
            {"x": 105, "y": 200},
        ]
        
        left, right = split_columns(regions)
        
        assert len(left) == 3
        assert len(right) == 0
    
    def test_empty_input(self):
        """Test empty input."""
        left, right = split_columns([])
        assert len(left) == 0
        assert len(right) == 0
    
    def test_single_region(self):
        """Test single region."""
        regions = [{"x": 100, "y": 100}]
        left, right = split_columns(regions)
        assert len(left) == 1
        assert len(right) == 0


class TestPriceHitRate:
    """Tests for _price_hit_rate helper function."""
    
    def test_all_have_prices(self):
        """Test column where all regions have prices."""
        column = [
            {"price_value": 3.0},
            {"price_value": 3.5},
            {"price_value": 4.0},
        ]
        assert _price_hit_rate(column) == 1.0
    
    def test_no_prices(self):
        """Test column with no prices."""
        column = [
            {"price_value": None},
            {"price_value": None},
        ]
        assert _price_hit_rate(column) == 0.0
    
    def test_partial_prices(self):
        """Test column with some prices."""
        column = [
            {"price_value": 3.0},
            {"price_value": None},
            {"price_value": 4.0},
        ]
        assert _price_hit_rate(column) == pytest.approx(2/3)
    
    def test_empty_column(self):
        """Test empty column."""
        assert _price_hit_rate([]) == 0.0


class TestIdentifyNameAndPriceColumns:
    """Tests for identify_name_and_price_columns function."""
    
    def test_right_column_has_more_prices(self):
        """Test typical case where right column is prices."""
        left = [
            {"price_value": None},
            {"price_value": None},
        ]
        right = [
            {"price_value": 3.0},
            {"price_value": 3.5},
        ]
        
        names, prices = identify_name_and_price_columns(left, right)
        
        assert names == left
        assert prices == right
    
    def test_left_column_has_more_prices(self):
        """Test when left column actually has prices (reversed layout)."""
        left = [
            {"price_value": 3.0},
            {"price_value": 3.5},
        ]
        right = [
            {"price_value": None},
            {"price_value": None},
        ]
        
        names, prices = identify_name_and_price_columns(left, right)
        
        assert names == right
        assert prices == left
    
    def test_no_right_column(self):
        """Test when there's no right column (single column)."""
        left = [{"price_value": None}]
        right = []
        
        names, prices = identify_name_and_price_columns(left, right)
        
        assert names == left
        assert prices == []


class TestPairItems:
    """Tests for pair_items function."""
    
    def test_perfect_pairing(self):
        """Test perfect one-to-one pairing."""
        names = [
            {"text": "Espresso", "y": 100, "confidence": 0.95},
            {"text": "Cappuccino", "y": 150, "confidence": 0.92},
        ]
        prices = [
            {"text": "3.0", "y": 102, "confidence": 0.85, "price_value": 3.0, 
             "price_raw": "3.0", "price_ambiguous": False, "currency": "TND", "currency_source": "default"},
            {"text": "3.5", "y": 152, "confidence": 0.88, "price_value": 3.5,
             "price_raw": "3.5", "price_ambiguous": False, "currency": "TND", "currency_source": "default"},
        ]
        
        items, orphans = pair_items(names, prices, max_y_distance=60)
        
        assert len(items) == 2
        assert len(orphans) == 0
        
        assert items[0]["name"] == "Espresso"
        assert items[0]["price_value"] == 3.0
        assert not items[0]["is_category_header"]
        
        assert items[1]["name"] == "Cappuccino"
        assert items[1]["price_value"] == 3.5
    
    def test_category_header_no_price(self):
        """Test name with no nearby price becomes category header."""
        names = [
            {"text": "Coffee", "y": 50, "confidence": 0.95},
            {"text": "Espresso", "y": 150, "confidence": 0.92},
        ]
        prices = [
            {"text": "3.0", "y": 152, "confidence": 0.85, "price_value": 3.0,
             "price_raw": "3.0", "price_ambiguous": False, "currency": "TND", "currency_source": "default"},
        ]
        
        items, orphans = pair_items(names, prices, max_y_distance=60)
        
        assert len(items) == 2
        assert items[0]["name"] == "Coffee"
        assert items[0]["is_category_header"] is True
        assert items[0]["price_value"] is None
        
        assert items[1]["name"] == "Espresso"
        assert items[1]["is_category_header"] is False
        assert items[1]["price_value"] == 3.0
    
    def test_orphan_price(self):
        """Test price with no nearby name becomes orphan."""
        names = [
            {"text": "Espresso", "y": 100, "confidence": 0.95},
        ]
        prices = [
            {"text": "3.0", "y": 102, "confidence": 0.85, "price_value": 3.0,
             "price_raw": "3.0", "price_ambiguous": False, "currency": "TND", "currency_source": "default"},
            {"text": "5.0", "y": 300, "confidence": 0.85, "price_value": 5.0,
             "price_raw": "5.0", "price_ambiguous": False, "currency": "TND", "currency_source": "default"},
        ]
        
        items, orphans = pair_items(names, prices, max_y_distance=60)
        
        assert len(items) == 1
        assert len(orphans) == 1
        
        assert orphans[0]["price_value"] == 5.0
        assert orphans[0]["text"] == "5.0"
    
    def test_nearest_neighbor_pairing(self):
        """Test that nearest price is paired, not first available."""
        names = [
            {"text": "Item1", "y": 100, "confidence": 0.95},
            {"text": "Item2", "y": 200, "confidence": 0.95},
        ]
        prices = [
            {"text": "5.0", "y": 205, "confidence": 0.85, "price_value": 5.0,
             "price_raw": "5.0", "price_ambiguous": False, "currency": "TND", "currency_source": "default"},
            {"text": "3.0", "y": 105, "confidence": 0.85, "price_value": 3.0,
             "price_raw": "3.0", "price_ambiguous": False, "currency": "TND", "currency_source": "default"},
        ]
        
        items, orphans = pair_items(names, prices, max_y_distance=60)
        
        # Item1 should pair with 3.0 (closer), Item2 with 5.0
        assert items[0]["name"] == "Item1"
        assert items[0]["price_value"] == 3.0
        
        assert items[1]["name"] == "Item2"
        assert items[1]["price_value"] == 5.0
    
    def test_empty_inputs(self):
        """Test empty inputs."""
        items, orphans = pair_items([], [])
        assert len(items) == 0
        assert len(orphans) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
