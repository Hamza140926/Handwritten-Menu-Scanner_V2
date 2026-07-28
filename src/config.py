"""
Central configuration for the handwritten menu scanner pipeline.

All tunable parameters are defined here as dataclasses.
This makes the system configurable without code changes.

Usage:
    from config import get_config
    
    config = get_config()
    max_dim = config.preprocessing.max_dimension
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import json


# config.py lives in src/, so project root is one level up
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

@dataclass
class PreprocessingConfig:
    """Image preprocessing configuration."""
    
    # Image resizing
    max_dimension: int = 2000  # px - cap longer side for consistent processing
    
    # Skew correction
    max_skew_degrees: float = 15.0  # degrees - safety limit on rotation
    
    # Denoising
    denoise_strength: int = 7  # higher = more denoising, more blur
    denoise_template_window: int = 7
    denoise_search_window: int = 21
    
    # Contrast normalization (CLAHE)
    clahe_clip_limit: float = 2.5  # higher = more contrast enhancement
    clahe_tile_size: tuple = (8, 8)  # grid size for local contrast


@dataclass
class DetectionConfig:
    """Text region detection configuration."""
    
    # Model selection
    model_name: str = "PP-OCRv5_mobile_det"  # or "PP-OCRv5_server_det" for higher accuracy
    
    # Detection parameters
    min_box_area: int = 200  # px² - filter out tiny boxes (noise)
    batch_size: int = 1  # for detection (usually 1 for single images)
    
    # Detection strategy presets for interactive retry
    strategies: dict = None  # Will be set in __post_init__
    
    def __post_init__(self):
        if self.strategies is None:
            self.strategies = {
                "default": {
                    "model_name": "PP-OCRv5_mobile_det",
                    "min_box_area": 200,
                    "description": "Balanced - good for most menus"
                },
                "sensitive": {
                    "model_name": "PP-OCRv5_mobile_det",
                    "min_box_area": 100,
                    "description": "Sensitive - catch smaller/faint text"
                },
                "strict": {
                    "model_name": "PP-OCRv5_mobile_det",
                    "min_box_area": 400,
                    "description": "Strict - filter out small noise/artifacts"
                },
                "server": {
                    "model_name": "PP-OCRv5_server_det",
                    "min_box_area": 200,
                    "description": "High accuracy - larger model, slower"
                },
            }
    
    
    # Device selection
    device: str = "cpu"  # "cpu" or "gpu:0"
    enable_mkldnn: bool = False  # MKL-DNN optimization (disabled due to PaddlePaddle bug)
    
    # Reading order sorting
    row_tolerance: int = 20  # px - regions within this y-distance are same row


@dataclass
class RecognitionConfig:
    """Handwriting recognition configuration."""
    
    # Model selection
    model_checkpoint: str = str(MODELS_DIR / "trocr_menu_v1_digits_v3" / "checkpoints" / "checkpoint-765")    # Alternatives: "models/trocr_menu_v1_digits_v3/checkpoints/checkpoint-765"

    # models/trocr_menu_v1_digits_v3/checkpoints/checkpoint-765
    # "microsoft/trocr-base-handwritten" (pretrained base model)
    # "microsoft/trocr-large-handwritten" (more accurate, slower)
    # "models/trocr_menu_v1/checkpoints/checkpoint-769" (epoch 1)
    # "models/trocr_menu_v1/checkpoints/checkpoint-1538" (epoch 2 - current)
    
    # Processing parameters
    batch_size: int = 16  # Higher = faster but more GPU memory
    max_new_tokens: int = 32  # Max length of recognized text
    
    # Device selection
    device: str = "auto"  # "auto", "cpu", or "cuda"
    use_fp16_on_gpu: bool = True  # Half precision on GPU (faster, less memory)


@dataclass
class PostprocessingConfig:
    """Price extraction and currency resolution configuration."""
    
    # Price extraction (improved with word boundaries)
    price_pattern: str = r"\b\d{1,6}(?:[.,]\d{1,3})?\b"  # Word boundary, 1-6 digits, optional decimal
    min_reasonable_price: float = 0.01  # Minimum valid price
    max_reasonable_price: float = 99999.0  # Maximum valid price
    
    # Currency resolution
    currency_confidence_threshold: float = 0.85  # Min confidence to trust detected currency
    default_currency: str = "TND"  # Default currency for menus
    supported_currencies: set = None  # Will be {"TND", "EUR"}
    
    def __post_init__(self):
        if self.supported_currencies is None:
            self.supported_currencies = {"TND", "EUR"}


@dataclass
class PipelineConfig:
    """Menu assembly and item pairing configuration."""
    
    # Column splitting
    column_gap_factor: float = 2.0  # Gap must be this much bigger than avg to split
    
    # Item pairing
    max_y_distance: int = 60  # px - max vertical distance to pair name with price
    relative_y_distance: bool = False  # If True, make y_distance relative to image height
    
    # Quality thresholds
    min_confidence_warning: float = 0.5  # Warn if confidence below this
    max_orphan_price_ratio: float = 0.3  # Warn if >30% prices are orphaned


@dataclass
class ValidationConfig:
    """Input validation configuration."""
    
    # File size limits
    max_file_size: int = 50 * 1024 * 1024  # 50 MB
    min_file_size: int = 1024  # 1 KB
    
    # Image dimension limits
    max_dimension: int = 10000  # px
    min_dimension: int = 100  # px
    
    # Allowed formats
    allowed_extensions: set = None
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


@dataclass
class OptimizationConfig:
    """Performance optimization configuration."""
    
    # GPU settings
    enable_tf32: bool = True  # TensorFloat-32 on Ampere GPUs (2x speedup)
    enable_cudnn_benchmark: bool = True  # Auto-tune convolution algorithms
    
    # Memory management
    clear_cache_after_batch: bool = True
    
    # Warmup
    warmup_gpu_on_startup: bool = True


@dataclass
class Config:
    """Master configuration containing all subsystem configs."""
    
    preprocessing: PreprocessingConfig = None
    detection: DetectionConfig = None
    recognition: RecognitionConfig = None
    postprocessing: PostprocessingConfig = None
    pipeline: PipelineConfig = None
    validation: ValidationConfig = None
    optimization: OptimizationConfig = None
    
    def __post_init__(self):
        # Initialize sub-configs if not provided
        if self.preprocessing is None:
            self.preprocessing = PreprocessingConfig()
        if self.detection is None:
            self.detection = DetectionConfig()
        if self.recognition is None:
            self.recognition = RecognitionConfig()
        if self.postprocessing is None:
            self.postprocessing = PostprocessingConfig()
        if self.pipeline is None:
            self.pipeline = PipelineConfig()
        if self.validation is None:
            self.validation = ValidationConfig()
        if self.optimization is None:
            self.optimization = OptimizationConfig()
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, dict):
                result[key] = value
            else:
                result[key] = asdict(value) if hasattr(value, '__dataclass_fields__') else value
        return result
    
    def save(self, path: Path):
        """Save configuration to JSON file."""
        path = Path(path)
        config_dict = self.to_dict()
        
        # Convert sets to lists for JSON serialization
        def convert_sets(obj):
            if isinstance(obj, dict):
                return {k: convert_sets(v) for k, v in obj.items()}
            elif isinstance(obj, set):
                return sorted(list(obj))
            elif isinstance(obj, tuple):
                return list(obj)
            return obj
        
        config_dict = convert_sets(config_dict)
        path.write_text(json.dumps(config_dict, indent=2))
    
    @classmethod
    def load(cls, path: Path) -> 'Config':
        """Load configuration from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        data = json.loads(path.read_text())
        
        # Reconstruct nested dataclasses
        config = cls()
        if 'preprocessing' in data:
            config.preprocessing = PreprocessingConfig(**data['preprocessing'])
        if 'detection' in data:
            config.detection = DetectionConfig(**data['detection'])
        if 'recognition' in data:
            config.recognition = RecognitionConfig(**data['recognition'])
        if 'postprocessing' in data:
            data['postprocessing']['supported_currencies'] = set(data['postprocessing'].get('supported_currencies', ['TND', 'EUR']))
            config.postprocessing = PostprocessingConfig(**data['postprocessing'])
        if 'pipeline' in data:
            config.pipeline = PipelineConfig(**data['pipeline'])
        if 'validation' in data:
            data['validation']['allowed_extensions'] = set(data['validation'].get('allowed_extensions', ['.jpg', '.jpeg', '.png']))
            config.validation = ValidationConfig(**data['validation'])
        if 'optimization' in data:
            config.optimization = OptimizationConfig(**data['optimization'])
        
        return config


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns the default configuration on first call.
    Use set_config() or load_config() to customize.
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config):
    """Set a custom configuration globally."""
    global _config
    _config = config


def load_config(path: Path):
    """Load configuration from file and set it globally."""
    config = Config.load(path)
    set_config(config)
    return config


def reset_config():
    """Reset to default configuration."""
    global _config
    _config = Config()


if __name__ == "__main__":
    # Demo: create and save default config
    config = Config()
    
    print("Default Configuration:")
    print("=" * 70)
    print(f"Preprocessing max_dimension: {config.preprocessing.max_dimension}")
    print(f"Detection model: {config.detection.model_name}")
    print(f"Recognition batch_size: {config.recognition.batch_size}")
    print(f"Postprocessing default_currency: {config.postprocessing.default_currency}")
    print(f"Pipeline max_y_distance: {config.pipeline.max_y_distance}")
    print(f"Validation max_file_size: {config.validation.max_file_size / 1024 / 1024}MB")
    
    # Save example config
    example_path = Path("config.example.json")
    config.save(example_path)
    print(f"\n✅ Example config saved to: {example_path}")
    
    # Test load
    loaded = Config.load(example_path)
    print(f"✅ Config loaded successfully")
    print(f"   Loaded max_dimension: {loaded.preprocessing.max_dimension}")
