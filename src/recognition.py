"""
Handwriting recognition for detected menu text regions.

Reads text out of each cropped region detection.py found. Per the
architecture spec (docs/handwritten-menu-scanner-spec.md, §5.3), one
unified model reads both item names and prices — digits are just part of
the same character vocabulary as letters, so there's no separate
digit/text model or routing step to decide which crop goes where.

Model: microsoft/trocr-base-handwritten, via Hugging Face `transformers`.
    - "base", not "large" — fits inference (and later, light fine-tuning)
      in 4GB VRAM.
    - Pretrained weights only, no fine-tuning yet. This gives a real
      baseline before investing in training (per spec §6: don't
      over-invest in fine-tuning before a working eval harness exists).

Usage:
    from recognition import recognize_regions

    results = recognize_regions(regions)
    # regions -> the list of dicts produced by detection.detect_text_regions
    # results -> list of dicts, one per region, each with:
    #   "text":       recognized string
    #   "confidence": float in [0, 1], mean token-level probability
    #   "box":        pass-through of the region's original box
    #   "y", "x":     pass-through, for reading-order sorting downstream

This feeds into postprocess.py (price/currency parsing).
"""

import cv2
import numpy as np
import torch
import threading
from logging_config import get_logger
from exceptions import RecognitionError
from config import get_config

logger = get_logger(__name__)

_processor = None  # lazy-loaded singletons so weights load once, not per call
_model = None
_device = None
_model_lock = threading.Lock()  # thread-safe initialization


def reset_recognition_model():
    """Release the cached model so the next scan loads the active pointer."""
    global _processor, _model, _device
    with _model_lock:
        _processor = None
        _model = None
        _device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _get_model():
    """Load the TrOCR processor + model once and reuse across calls.

    Runs on GPU if available (fp16, to fit 4GB VRAM per the spec's
    compute plan), otherwise falls back to CPU fp32.

    Thread-safe: Multiple simultaneous calls will wait for initialization
    to complete rather than creating duplicate model instances.
    
    Raises:
        RecognitionError: If model loading fails
    """
    global _processor, _model, _device
    if _model is None:
        with _model_lock:
            # Double-check locking pattern: another thread might have
            # initialized while we were waiting for the lock
            if _model is None:
                try:
                    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                    import os
                    
                    cfg = get_config().recognition
                    
                    # Auto device selection: "auto" means try CUDA first, fallback to CPU
                    if cfg.device == "auto":
                        _device = "cuda" if torch.cuda.is_available() else "cpu"
                    else:
                        _device = cfg.device
                    
                    logger.info("Loading TrOCR model", extra={"checkpoint": cfg.model_checkpoint, "device": _device})

                    # Check if this is a local checkpoint path (missing processor files)
                    # Checkpoints only save model weights, not processor config
                    is_local_checkpoint = (
                        os.path.exists(cfg.model_checkpoint) and 
                        os.path.isdir(cfg.model_checkpoint) and
                        not os.path.exists(os.path.join(cfg.model_checkpoint, "preprocessor_config.json"))
                    )
                    
                    if is_local_checkpoint:
                        # Load processor from base model, weights from checkpoint
                        base_model = "microsoft/trocr-base-handwritten"
                        logger.info("Loading processor from base model", extra={"base": base_model})
                        _processor = TrOCRProcessor.from_pretrained(base_model)
                        _model = VisionEncoderDecoderModel.from_pretrained(cfg.model_checkpoint)
                    else:
                        # Standard loading (works for HuggingFace models and full saves)
                        _processor = TrOCRProcessor.from_pretrained(cfg.model_checkpoint)
                        _model = VisionEncoderDecoderModel.from_pretrained(cfg.model_checkpoint)

                    if _device == "cuda" and cfg.use_fp16_on_gpu:
                        _model = _model.half().to(_device)
                    else:
                        _model = _model.to(_device)
                    _model.eval()

                    logger.info("TrOCR model loaded successfully", extra={"checkpoint": cfg.model_checkpoint, "device": _device})
                
                except Exception as e:
                    logger.exception("Failed to load TrOCR model")
                    raise RecognitionError(f"Failed to initialize recognition model: {e}") from e

    return _processor, _model, _device


def _bgr_to_pil(crop: np.ndarray):
    """Convert an OpenCV BGR crop (as produced by detection.py) into the
    RGB PIL image TrOCR's processor expects."""
    from PIL import Image

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _mean_token_confidence(scores, sequences, pad_token_id) -> list:
    """Compute a single confidence score per generated sequence, averaged
    over the actual (non-padding) generated tokens.

    `scores` is the tuple of per-step logits returned by generate() with
    output_scores=True — one tensor of shape (batch, vocab) per decoding
    step. We take the softmax probability of the token that was actually
    picked at each step, then average those probabilities across the
    tokens that make up each real (unpadded) sequence.
    """
    if not scores:
        return [1.0] * sequences.shape[0]

    # scores[t]: (batch, vocab) logits for decoding step t.
    # sequences[:, 0] is the BOS/decoder-start token, so generated tokens
    # start at sequences[:, 1:], aligned 1:1 with scores.
    # BUT: scores length might be shorter than sequences if generation stopped early
    step_probs = []
    for t, step_logits in enumerate(scores):
        probs = torch.softmax(step_logits.float(), dim=-1)
        # Bounds check: t+1 must be valid index in sequences
        if t + 1 >= sequences.shape[1]:
            break
        chosen_ids = sequences[:, t + 1]
        chosen_probs = probs.gather(1, chosen_ids.unsqueeze(1)).squeeze(1)
        step_probs.append(chosen_probs)

    if not step_probs:
        return [1.0] * sequences.shape[0]

    step_probs = torch.stack(step_probs, dim=1)  # (batch, num_steps)
    generated = sequences[:, 1:1 + step_probs.shape[1]]
    mask = (generated != pad_token_id).float()

    # Avoid divide-by-zero for any empty/all-pad sequence.
    token_counts = mask.sum(dim=1).clamp(min=1.0)
    confidences = (step_probs * mask).sum(dim=1) / token_counts
    return confidences.tolist()


def recognize_regions(regions: list, batch_size: int = None) -> list:
    """Run TrOCR over each detected region and return recognized text with
    a confidence score.

    Args:
        regions: list of dicts from detection.detect_text_regions, each
            with at least a "crop" (BGR image) key.
        batch_size: how many crops to feed through the model at once. 
            If None, uses config value. Increase for more VRAM, decrease if OOM.

    Returns:
        List of dicts, one per input region, each with:
            "text":       recognized string (stripped)
            "confidence": float in [0, 1]
            "box":        pass-through from the input region, if present
            "y", "x":     pass-through from the input region, if present
            
    Raises:
        RecognitionError: If recognition fails
    """
    if not regions:
        logger.warning("No regions to recognize")
        return []

    try:
        cfg = get_config().recognition
        if batch_size is None:
            batch_size = cfg.batch_size
        
        logger.debug("Starting recognition", extra={"region_count": len(regions), "batch_size": batch_size})
        processor, model, device = _get_model()
        results = [None] * len(regions)

        for start in range(0, len(regions), batch_size):
            batch = regions[start:start + batch_size]
            images = [_bgr_to_pil(r["crop"]) for r in batch]

            pixel_values = processor(images=images, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            if device == "cuda":
                pixel_values = pixel_values.half()

            with torch.no_grad():
                generated = model.generate(
                    pixel_values,
                    max_new_tokens=cfg.max_new_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            texts = processor.batch_decode(generated.sequences, skip_special_tokens=True)
            confidences = _mean_token_confidence(
                generated.scores, generated.sequences, processor.tokenizer.pad_token_id
            )

            for i, region in enumerate(batch):
                results[start + i] = {
                    "text": texts[i].strip(),
                    "confidence": float(confidences[i]),
                    "box": region.get("box"),
                    "y": region.get("y"),
                    "x": region.get("x"),
                }
            
            # Memory cleanup: free intermediate tensors after each batch
            del pixel_values, generated, texts, confidences, images
            if device == "cuda":
                torch.cuda.empty_cache()
        
        logger.info("Recognition complete", extra={"regions_processed": len(regions)})
        return results
    
    except RecognitionError:
        raise
    except Exception as e:
        logger.exception("Recognition failed")
        raise RecognitionError(f"Recognition pipeline failed: {e}") from e


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocessing import preprocess_image
    from detection import detect_text_regions
    from logging_config import setup_logging
    
    setup_logging(level="INFO")

    if len(sys.argv) != 2:
        logger.error("Missing image path argument")
        print("Usage: python recognition.py <path_to_image>")
        sys.exit(1)

    prep = preprocess_image(sys.argv[1])
    regions = detect_text_regions(prep["image"])

    results = recognize_regions(regions)
    for i, r in enumerate(results):
        logger.info(f"Region {i:02d}", extra={"confidence": r['confidence'], "text": r['text']})
