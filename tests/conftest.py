"""
Pytest configuration and shared fixtures.

Fixtures defined here are available to all test files.
"""

import pytest
import sys
from pathlib import Path

# Add src to path so tests can import modules
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_image_path():
    """Path to a sample menu image for testing."""
    # TODO: Add actual sample image to tests/fixtures/
    return "tests/fixtures/sample_menu.jpg"


@pytest.fixture
def sample_price_strings():
    """Sample recognized text strings for price extraction testing."""
    return [
        "12.50",
        "12,50",
        "3.500",  # Tunisian style
        "Espresso 3.0",
        "3, 3, 3.",  # Ambiguous/noisy
        "Cappuccino",  # No price
        "€5.00",
        "12.500 DT",
    ]


@pytest.fixture
def sample_regions():
    """Sample detected regions for testing pairing logic."""
    return [
        # Left column (names)
        {"text": "Espresso", "x": 100, "y": 100, "confidence": 0.95, "price_value": None},
        {"text": "Cappuccino", "x": 100, "y": 150, "confidence": 0.92, "price_value": None},
        {"text": "Mocha", "x": 100, "y": 200, "confidence": 0.89, "price_value": None},
        # Right column (prices)
        {"text": "3.0", "x": 300, "y": 102, "confidence": 0.85, "price_value": 3.0},
        {"text": "3.5", "x": 300, "y": 152, "confidence": 0.88, "price_value": 3.5},
        {"text": "4.0", "x": 300, "y": 198, "confidence": 0.90, "price_value": 4.0},
    ]
