# Recognition Step — Status Report

**Component:** `src/recognition.py`
**Stage:** Handwriting recognition (spec §5.3)
**Status:** Working baseline, running on GPU, no fine-tuning yet
**Last updated:** July 2026

---

## 1. Where we are

`detection.py` (previous step) is done and confirmed working — it finds
text regions on a preprocessed menu photo and outputs cropped image
regions in reading order.

`recognition.py` is now implemented and confirmed working end-to-end on
a real photographed Tunisian menu (not a synthetic test image). It:

- Loads **`microsoft/trocr-base-handwritten`** via Hugging Face
  `transformers` — one unified model reads both item names and prices,
  per the architecture decision in spec §5.3 (no separate digit model).
- Takes the list of cropped regions from `detection.py` and returns, per
  region: recognized `text`, a `confidence` score, and the original
  `box`/position data passed through for downstream use.
- Runs on GPU (fp16) when available, with a CPU fallback.
- Uses pretrained weights only — no fine-tuning, per the spec's decision
  to get a real baseline before investing in training.

This satisfies the first bullet of the spec's rollout sequence (§9.1):
preprocessing + detection + recognition working as a local pipeline.
`postprocess.py` (price/currency parsing) is the next step, not yet
started.

---

## 2. Issues encountered and how they were fixed

### 2.1 Missing `sentencepiece` dependency

**Symptom:**
```
ValueError: Couldn't instantiate the backend tokenizer from one of:
...
You need to have sentencepiece or tiktoken installed to convert a slow
tokenizer to a fast one.
```

**Fix:** `pip install sentencepiece`, and added it to `requirements.txt`.
This is a real, permanent dependency of TrOCR's tokenizer that was
missing from the original requirements list — not a one-off local
issue.

### 2.2 Model loaded on CPU despite having a GPU

**Symptom:** log line read `Loading microsoft/trocr-base-handwritten on
cpu...` despite the dev machine having an RTX 3050.

**Root cause:** `torch.cuda.is_available()` was returning `False`
because the installed `torch` build was CPU-only. On Windows, a plain
`pip install torch` does **not** include CUDA support by default (unlike
Linux) — you have to install from PyTorch's dedicated CUDA index.

**Fix:** Uninstalled the CPU build and reinstalled from the CUDA 13.0
index, matching the machine's driver (`nvidia-smi` reported CUDA 13.2):
```
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu130
```
An earlier attempt at `cu121` failed outright (`no versions found`) —
that index doesn't publish wheels for Python 3.13, which is what the
dev environment is running. `cu130` does. Confirmed after reinstall:
`torch.cuda.is_available()` → `True`, and `recognition.py` picked up
`cuda` automatically (no code change needed — device selection was
already conditional on this check).

### 2.3 Silent generation-length truncation risk

**Symptom:** a `transformers` warning surfaced during a real run:
```
Using the model-agnostic default `max_length` (=20) to control the
generation length.
```
Not an error on the current test menu (all lines short enough), but a
latent risk: this default would silently truncate any line combining a
longer item name with a price (e.g. `"Couscous 12.500"`), which the spec
explicitly calls out as a case the model needs to handle in one pass.

**Fix:** set `max_new_tokens=32` explicitly in the `generate()` call so
longer combined lines aren't cut off.

---

## 3. Baseline recognition quality (first real-menu run)

Ran against 14 detected regions from a real handwritten Tunisian coffee
menu photo. Representative results:

| Region | Confidence | Recognized text | Note |
|---|---|---|---|
| cappuccino | 0.96 | `cappuccino` | correct |
| mocha | 0.99 | `mocha` | correct |
| hot tea | 0.98 | `hot tea .` | correct (minor trailing artifact) |
| Double Espresso | 0.81 | `Double Espresso` | correct |
| price region | 0.59 | `3, 3, 3.` | garbled |
| price region | 0.53 | `ly , 4 ,4st` | garbled |
| price region | 0.73 | `4 DE` | garbled currency attempt |

**Takeaways:**

- **Item-name lines are recognized well.** This is consistent with
  TrOCR-handwritten's pretraining on IAM (general English handwriting).
- **Price/number regions are the weak point.** This lines up with a
  known limitation of IAM-pretrained handwriting OCR — it's trained
  mostly on words, not isolated digit strings — and it retroactively
  validates the spec's §5.4/§5.5 decision to **not** trust OCR-read
  currency letters (`DT`, `dt`, etc.) and instead extract only the
  numeric value via regex, with currency set by an owner-level toggle.
  The garbled attempts at currency abbreviations above (`DE`, `ft`,
  `gdt`) are exactly the failure mode that design choice was meant to
  route around.
- **Confidence doesn't cleanly separate correct from incorrect output
  yet.** Some wrong price reads scored similar to or higher than other
  wrong reads, rather than clustering low. This doesn't block moving
  forward, but it means the review-UI confidence highlighting (§5.6)
  may need real usage data (§8) before it's a reliable triage signal —
  worth tracking once correction logging is in place, not something to
  solve now on a single test image.

This overall result is in line with the spec's realistic accuracy target
(§4): strong on words, weaker on numbers, "good enough for glance-and-fix"
rather than exact-match — which is the intended bar, not a shortfall.

---

## 4. Next step

Build `postprocess.py`: regex-based price/number extraction from each
recognized string, tolerant of both `.` and `,` as decimal separators,
ignoring currency letters entirely per §5.4–5.5. Not yet started.
