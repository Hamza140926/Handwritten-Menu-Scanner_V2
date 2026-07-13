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

MODEL_CHECKPOINT = "microsoft/trocr-base-handwritten"

# Per the spec's postmortem lesson (§6): an earlier attempt got burned by
# an eval harness that silently wasn't loading the checkpoint it thought
# it was. Always log/print exactly which checkpoint is in use before
# trusting output, rather than assuming.
print(f"[recognition] Using model checkpoint: {MODEL_CHECKPOINT}")

_processor = None  # lazy-loaded singletons so weights load once, not per call
_model = None
_device = None


def _get_model():
    """Load the TrOCR processor + model once and reuse across calls.

    Runs on GPU if available (fp16, to fit 4GB VRAM per the spec's
    compute plan), otherwise falls back to CPU fp32.
    """
    global _processor, _model, _device
    if _model is None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[recognition] Loading {MODEL_CHECKPOINT} on {_device}...")

        _processor = TrOCRProcessor.from_pretrained(MODEL_CHECKPOINT)
        _model = VisionEncoderDecoderModel.from_pretrained(MODEL_CHECKPOINT)

        if _device == "cuda":
            _model = _model.half().to(_device)
        else:
            _model = _model.to(_device)
        _model.eval()

        print(f"[recognition] Model loaded. Checkpoint confirmed: {_model.name_or_path}")

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
    step_probs = []
    for t, step_logits in enumerate(scores):
        probs = torch.softmax(step_logits.float(), dim=-1)
        chosen_ids = sequences[:, t + 1]
        chosen_probs = probs.gather(1, chosen_ids.unsqueeze(1)).squeeze(1)
        step_probs.append(chosen_probs)

    step_probs = torch.stack(step_probs, dim=1)  # (batch, num_steps)
    generated = sequences[:, 1:1 + step_probs.shape[1]]
    mask = (generated != pad_token_id).float()

    # Avoid divide-by-zero for any empty/all-pad sequence.
    token_counts = mask.sum(dim=1).clamp(min=1.0)
    confidences = (step_probs * mask).sum(dim=1) / token_counts
    return confidences.tolist()


def recognize_regions(regions: list, batch_size: int = 8) -> list:
    """Run TrOCR over each detected region and return recognized text with
    a confidence score.

    Args:
        regions: list of dicts from detection.detect_text_regions, each
            with at least a "crop" (BGR image) key.
        batch_size: how many crops to feed through the model at once. TrOCR
            base fits several crops per batch even on 4GB VRAM; lower this
            if you hit an out-of-memory error.

    Returns:
        List of dicts, one per input region, each with:
            "text":       recognized string (stripped)
            "confidence": float in [0, 1]
            "box":        pass-through from the input region, if present
            "y", "x":     pass-through from the input region, if present
    """
    if not regions:
        return []

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
                max_new_tokens=32,  # explicit, so longer lines (e.g. a long
                                    # item name plus a price, like "Couscous
                                    # 12.500") don't get silently truncated
                                    # at the model's default of 20
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

    return results


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocessing import preprocess_image
    from detection import detect_text_regions

    if len(sys.argv) != 2:
        print("Usage: python recognition.py <path_to_image>")
        sys.exit(1)

    prep = preprocess_image(sys.argv[1])
    regions = detect_text_regions(prep["image"])
    print(f"Detected {len(regions)} text regions")

    results = recognize_regions(regions)
    for i, r in enumerate(results):
        print(f"[{i:02d}] conf={r['confidence']:.2f}  text={r['text']!r}")