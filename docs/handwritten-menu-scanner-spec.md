# Handwritten Menu Scanner — Project Spec
**Feature for: Scan2See app**
**Status:** Planning stage
**Last updated:** July 2026

---

## 1. Goal

Give business owners (coffee shops, restaurants, spas, hotels, service
providers) the ability to scan a **handwritten paper menu** and get back a
**digital, editable version** they can tweak inside the app — instead of
typing every item and price manually from scratch.

This is **not** a fully-automated, zero-review pipeline. The AI output is a
**draft**. The owner reviews and corrects it before it's saved. The goal is
to save time and reduce typing, not to achieve perfect unattended
extraction.

---

## 2. Why this matters (design philosophy)

Early exploration of this problem chased **98–99% exact-match accuracy**,
which is not realistic for general handwriting recognition — not even for
commercial-grade systems (Google/Microsoft handwriting OCR tops out around
90–95% *character*-level accuracy on clean handwriting; *exact full-string*
match on messy real-world menus is a much harder bar).

Once reframed as **"draft + owner edits"**, the bar drops to something
achievable: get enough of the menu right, with clear visual cues on what to
double check, that fixing it is faster than typing it from zero.

**Core principle: build for correction, not perfection.**

---

## 3. Real-world constraints

| Constraint | Detail |
|---|---|
| Handwriting | Fully general — any owner/waiter, any handwriting style. No single consistent writer to adapt to. |
| Layout | Free-form / messy. Not a clean template or lined-notebook structure. |
| Menu types | Coffee shops, restaurants, drinks menus, spas, hotels, general services |
| Currency | 70–80% Tunisian Dinar, written inconsistently: `DT`, `dt`, `TND`, `D`, `d`, or **no currency marker at all** (common in long menus). Remaining menus: Euro, sometimes with `€` symbol, sometimes unspecified. |
| Labeled training data | **None available at start.** No existing dataset combines handwritten text + digits in a menu-like format. |
| Compute | Local laptop, RTX 3050, **4GB VRAM**. Google Colab free tier as backup for heavier jobs. |
| Budget | **$0.** No paid APIs, no cloud GPU rental, no international card available for payment. |
| Correction step | Owner can always edit the scanned result inside the app before saving. This is a first-class part of the flow, not a fallback. |

---

## 4. Accuracy target (realistic)

- **Not** 98–99% exact match — not achievable with general handwriting,
  messy layout, and no fine-tuning data.
- Realistic baseline expectation (pretrained model, no fine-tuning, general
  handwriting): roughly **60–80% exact line accuracy**, higher at the
  character level.
- Practical target for the feature to feel useful: **most fields close
  enough that a glance-and-fix beats typing from scratch.**
- Accuracy should improve over time via real usage data (see §8).

---

## 5. Architecture

### High-level pipeline
```
Photo of menu
   ↓
Preprocessing (deskew, contrast normalization)
   ↓
Text region detection (finds each line/item on the page)
   ↓
Handwriting recognition (single unified model — reads both text and digits)
   ↓
Post-processing (regex price extraction, currency resolution)
   ↓
Confidence scoring → drives review UI
   ↓
Owner review & correction (in-app)
   ↓
Saved to database
   ↓
Correction logged → future training data (see §8)
```

### 5.1 Preprocessing
- Deskew the photo
- Normalize contrast / lighting (menus photographed on paper often have
  shadows or uneven lighting — fixing this before OCR reduces a lot of
  garbage output for free)

### 5.2 Text region detection
- Layout is free-form/messy → **skip classical row-detection**
  (projection profiles / contour-based line splitting), it assumes
  structure that won't reliably exist here.
- Use a general text-detection model instead:
  - **PaddleOCR's detector (DBNet)**, or
  - **CRAFT text detector**
- Both are free, run locally, and are light enough for 4GB VRAM at
  inference time.
- Output: cropped image regions, one per detected text line/item.

### 5.3 Handwriting recognition
- **One unified model reads both item names and prices** — not two
  separate models merged together. Digits are just part of the same
  character vocabulary as letters; a single sequence model handles both in
  the same pass, including mixed lines like `Couscous 12.500`.
- Model: **`microsoft/trocr-base-handwritten`**
  - Not `-large` — base (~334M params) fits inference and light fine-tuning
    within 4GB VRAM (small batch size, fp16). Large would need Colab even
    for fine-tuning.
- Framework: **Hugging Face `transformers`** — well documented, TrOCR
  ships pretrained through it directly, runs fine locally or on Colab free
  tier. No need for a heavier/custom framework.
- **Start with the pretrained model, no fine-tuning.** It's already trained
  on general handwriting corpora (IAM etc.), which matches the "anyone,
  general handwriting" requirement better than a narrow fine-tune with no
  real data would anyway.

### 5.4 Post-processing — price extraction
- Regex-based extraction of the numeric value from each recognized string,
  tolerant of both `.` and `,` as decimal separators (both appear in
  Tunisian handwriting).
- **Do not rely on OCR to correctly read currency letters** (`DT` vs `dt`
  vs `D` vs `d` vs blank) — handwriting recognition on 1–2 letter
  abbreviations is unreliable and it's unnecessary given §5.5.

### 5.5 Currency resolution (key simplification)
- Ask the owner to set a **default currency for the whole menu** (TND or
  EUR) — one toggle, set once, not per item.
- Use OCR-detected currency symbols (e.g. a clearly recognized `€`) only as
  an **override hint** when confidence is high, not as the primary source
  of truth.
- This removes the hardest, most error-prone part of the original
  approach (disambiguating messy currency abbreviations) and turns price
  recognition into a much simpler "extract a number" task.

### 5.6 Confidence scoring & review UI
- TrOCR provides token-level confidence scores — surface these.
- In the review screen, highlight low-confidence fields (e.g. faint yellow
  background) so the owner's attention goes where it's actually needed,
  instead of forcing an equal-effort review of every single line.
- Every field remains editable regardless of confidence.

### 5.7 Owner correction step (core, not fallback)
- After scan, the owner sees the extracted menu (name + price per item)
  in an editable table/list.
- They fix any mistakes before saving.
- This step is what makes a realistic (not 98–99%) accuracy level still
  genuinely useful in production.

---

## 6. What NOT to do (lessons from earlier attempts)

- **Don't train two separate models (digits-only + text-only) and try to
  merge them.** A single sequence model with a combined character
  vocabulary (letters + digits + punctuation) handles mixed content in one
  pass and avoids a fragile routing step to decide which model handles
  which crop.
- **Don't over-invest in synthetic-data fine-tuning before a working
  eval harness exists.** An earlier attempt fine-tuned a price-recognition
  model on synthetic font-rendered data across multiple rounds and plateaued
  at 8% exact-match accuracy no matter what was changed. Deeper
  investigation pointed to the evaluation setup — not the training data —
  as the likely actual bottleneck (evidence: an undertrained checkpoint and
  a fully-trained checkpoint scored *identically*, and "fixed" repetition
  bugs reappeared later, both signs the eval may not have been loading the
  intended model/checkpoint). **Lesson: build and sanity-check the
  evaluation harness first**, and always print/log exactly which model
  checkpoint is being evaluated, before drawing conclusions about model or
  data quality.
- **Don't chase exact-match accuracy as the success metric** given the
  owner-correction workflow. A wrong digit the owner fixes in two seconds
  is an acceptable outcome, not a failure.
- **Don't block launch on collecting a real handwriting dataset upfront.**
  See §8 — real data can be gathered as a natural byproduct of usage
  instead.

---

## 7. Compute plan

- **Local (RTX 3050, 4GB VRAM):**
  - Inference: TrOCR-base + PaddleOCR/CRAFT detector — feasible locally.
  - Light fine-tuning (small batch size, fp16, base model only) — feasible
    but slow.
- **Google Colab (free tier):**
  - Backup for any heavier fine-tuning runs or larger batch experiments
    that don't fit in 4GB VRAM.
- **No paid infrastructure** — not in scope, not budgeted, not available.

---

## 8. Turning real usage into training data (the actual dataset solution)

The original blocker was "no dataset exists that combines handwritten text
and digits in a menu-like format." Rather than manually collecting and
labeling one upfront:

- **Every correction an owner makes after a scan is a free, real, labeled
  training example** — real handwriting, real menu content, real currency
  quirks, zero manual labeling effort required from the dev side.
- Log scans in a structure that captures, per field:
  - the original cropped image region
  - the model's raw prediction
  - the owner's corrected value (if changed)
  - confidence score at prediction time
- After a few weeks of real usage, this log becomes a genuine
  domain-specific fine-tuning dataset — solving the original "no dataset"
  problem organically, using real data instead of synthetic fonts.
- Fine-tuning on this real data (even a modest amount — tens to low
  hundreds of corrected samples) is expected to meaningfully outperform
  synthetic-only fine-tuning, based on the earlier postmortem's own
  findings.

---

## 9. Rollout sequence

1. Build preprocessing + text detection + TrOCR inference + regex price
   parsing as a working local pipeline (no fine-tuning).
2. Build the review/correction UI (confidence highlighting + edit flow) —
   this is core to the feature, not an add-on.
3. Set up correction logging (per §8) from day one, even before there's
   enough data to do anything with it.
4. Ship to real owners with the "scan → review → edit → save" flow.
5. After sufficient real usage data accumulates, evaluate whether
   fine-tuning on real corrections meaningfully improves accuracy, using a
   proper evaluation harness (checkpoint path logged, sanity-checked
   before trusting results) from the start this time.

---

## 10. Open questions / future decisions

- Exact UI treatment for confidence highlighting (color thresholds, per
  field vs per line).
- Whether text detection needs a custom-trained layout model eventually,
  or general detectors (PaddleOCR/CRAFT) remain sufficient as menu variety
  grows.
- At what volume of logged corrections fine-tuning becomes worthwhile.
- Whether per-item currency override is ever needed, or menu-level default
  remains sufficient in practice.
