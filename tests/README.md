# Test Suite

Automated tests for the handwritten menu scanner pipeline.

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_postprocess.py -v

# Run specific test
pytest tests/test_postprocess.py::TestPriceExtraction::test_decimal_dot -v

# Run with coverage report
pytest --cov=src --cov-report=html

# Run only fast tests (exclude slow integration tests)
pytest -m "not slow"
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_postprocess.py      # Price extraction tests (COMPLETE)
├── test_validation.py       # Input validation tests (COMPLETE)
├── test_pipeline.py         # Menu assembly tests (COMPLETE)
├── test_preprocessing.py    # Image preprocessing tests (TODO)
├── test_detection.py        # Text detection tests (TODO)
├── test_recognition.py      # Handwriting recognition tests (TODO)
└── integration/             # Full pipeline integration tests (TODO)
```

## Test Coverage

Current coverage:
- ✅ **postprocess.py**: ~90% (price extraction, currency resolution)
- ✅ **validation.py**: ~80% (input validation, security checks)
- ✅ **pipeline.py**: ~70% (column splitting, item pairing)
- ⚠️ **preprocessing.py**: 0% (TODO)
- ⚠️ **detection.py**: 0% (TODO - needs mock PaddleOCR)
- ⚠️ **recognition.py**: 0% (TODO - needs mock TrOCR)

## Writing New Tests

1. Add test file in `tests/` following naming: `test_<module>.py`
2. Use fixtures from `conftest.py` where possible
3. Group related tests in classes: `class TestFeatureName:`
4. Name tests descriptively: `test_feature_with_specific_condition`
5. Use pytest markers for slow/integration tests

Example:
```python
import pytest
from mymodule import my_function

class TestMyFunction:
    def test_normal_case(self):
        result = my_function("input")
        assert result == "expected"
    
    @pytest.mark.slow
    def test_with_large_data(self):
        # Slow test
        pass
```

## Continuous Integration

Tests should run automatically on:
- Every commit (pre-commit hook)
- Every pull request
- Before deployment

Target: >80% code coverage before production release.
