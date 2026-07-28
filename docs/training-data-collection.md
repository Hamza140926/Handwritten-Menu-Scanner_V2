# Training data collection

The live scanner now records the same kind of examples consumed by
`training/train_synthetic.py`: one deskewed OCR crop plus its owner-corrected
text label. Name and price regions are stored separately so each label remains
paired with the exact pixels that TrOCR recognized.

## One-time setup

1. Create a free Cloudinary account.
2. Copy `.env.example` to `.env`.
3. Put the `CLOUDINARY_URL` from the Cloudinary dashboard in `.env`:

   ```dotenv
   CLOUDINARY_URL=cloudinary://API_KEY:API_SECRET@CLOUD_NAME
   SCANTOSEE_MODEL_VERSION=trocr-menu-digits-v3-checkpoint-765
   ```

4. Install the scanner requirements in the Conda environment:

   ```powershell
   conda activate training
   pip install -r requirements.txt
   ```

5. Apply Symfony migrations:

   ```powershell
   cd ..\my_project_directory
   php bin/console doctrine:migrations:migrate
   ```

## Run locally

Start the OCR API from `handwritten-menu-scanner/src`:

```powershell
conda activate training
cd handwritten-menu-scanner\src
python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

Start Symfony from `my_project_directory`:

```powershell
symfony serve
```

`OCR_PIPELINE_URL` must remain `http://127.0.0.1:8001`.

## What gets collected

- The validated original menu image
- One crop for every OCR region sent to TrOCR
- Raw text, confidence, geometry, structural role and pairing IDs
- Model/detector version metadata
- The text confirmed or corrected by the owner
- Deleted predictions, retained as review outcome `deleted`

The existing Category/Item save continues as before. Data collection is
additive and is stored in `scan_capture` and `scan_region`.

If Cloudinary is missing or temporarily unavailable, scanning still works.
Those scans remain auditable in MySQL, but their crop URL is null and they
cannot be exported as image/text training examples.

## Export for training

After collecting and reviewing scans:

```powershell
cd my_project_directory
php bin/console app:export-ocr-training-data C:\tmp\scantosee-training-export
```

The command downloads reviewed crops and creates:

```text
scantosee-training-export/
├── manifest.csv
└── crops/
```

The manifest has the required `crop_path`, `text`, and `split` columns, plus
audit metadata. All crops from one source scan receive the same deterministic
split to reduce leakage between training and evaluation.

Train with the existing script:

```powershell
cd handwritten-menu-scanner
conda activate training
python training\train_synthetic.py `
  --dataset_dir C:\tmp\scantosee-training-export `
  --output_model_dir models\trocr_menu_real_v1 `
  --eval_test
```
