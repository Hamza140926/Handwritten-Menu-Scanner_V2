"""FastAPI wrapper around the menu-scanning pipeline (pipeline.py).

Exposes one synchronous endpoint: POST /scan-menu
    - accepts an uploaded image (+ optional currency form field)
    - runs run_pipeline() on it
    - returns the resulting menu JSON (items, orphan_prices, quality_metrics)

Debug image generation is disabled (debug_output_path=None) - this API
is for dev use for now, no need to write files nobody reads.

GPU warmup / throughput optimization is triggered once at server
startup (lifespan), NOT at import time - pipeline.py currently runs
these at module import, which is fine for the CLI but wrong for a
long-lived server process where import happens once anyway; making it
explicit here avoids relying on that implicit side effect.
"""

import os
import secrets
import tempfile
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from pipeline import run_pipeline
from validation import ValidationError
from exceptions import (
    PipelineError, PreprocessingError, DetectionError,
    RecognitionError, PostprocessingError, AssemblyError,
)
from optimization import warmup_gpu, optimize_for_throughput
from config import MODELS_DIR, get_config, reset_config
from recognition import reset_recognition_model
from logging_config import setup_logging, get_logger
from storage import delete_scan_assets

setup_logging(level="INFO")
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: warming up GPU / applying throughput optimizations")
    optimize_for_throughput()
    warmup_gpu()
    yield
    logger.info("Shutting down")


app = FastAPI(title="Scantosee OCR Pipeline", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model_checkpoint": get_config().recognition.model_checkpoint}


@app.post("/admin/reload-model", include_in_schema=False)
async def reload_model(x_cleanup_token: Optional[str] = Header(default=None)):
    """Reload the locally promoted model. Accessible only through Symfony."""
    expected = os.getenv("OCR_CLEANUP_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=503, detail="Model administration is not configured.")
    if not x_cleanup_token or not secrets.compare_digest(x_cleanup_token, expected):
        raise HTTPException(status_code=403, detail="Forbidden.")
    reset_recognition_model()
    reset_config()
    checkpoint = os.path.realpath(get_config().recognition.model_checkpoint)
    models_root = os.path.realpath(str(MODELS_DIR))
    if os.path.commonpath([checkpoint, models_root]) != models_root or not os.path.isdir(checkpoint):
        raise HTTPException(status_code=422, detail="The promoted checkpoint is invalid.")
    return {"status": "ready", "model_checkpoint": checkpoint}


@app.post("/scan-menu")
async def scan_menu(
    image: UploadFile = File(...),
    currency: Optional[str] = Form(default=None),
):
    """Accepts a menu photo, runs the OCR pipeline synchronously, and
    returns the structured menu as JSON.

    currency: optional, e.g. "TND" or "EUR". Falls back to the
    pipeline's configured default_currency when omitted - same
    fallback the CLI (pipeline.py's __main__ block) already uses.
    """
    default_currency = currency or get_config().postprocessing.default_currency

    suffix = os.path.splitext(image.filename or "")[1] or ".jpg"
    tmp_path = None
    scan_uuid = str(uuid.uuid4())
    keep_training_assets = False
    try:
        contents = await image.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        logger.info(
            "Processing upload",
            extra={"upload_filename": image.filename, "currency": default_currency},
        )

        menu = run_pipeline(
            tmp_path,
            default_currency=default_currency,
            debug_output_path=None,
            scan_uuid=scan_uuid,
        )
        keep_training_assets = bool(menu.get("regions"))
        return menu

    except ValidationError as e:
        logger.warning("Validation failed", extra={"error": str(e)})
        return JSONResponse(
            status_code=400,
            content={"error": "ValidationError", "message": str(e), "items": []},
        )

    except (PreprocessingError, DetectionError, RecognitionError, PostprocessingError) as e:
        logger.error(f"{type(e).__name__} while processing upload", exc_info=True)
        return JSONResponse(
            status_code=422,
            content={"error": type(e).__name__, "message": str(e), "items": []},
        )

    except (AssemblyError, PipelineError) as e:
        logger.error(f"{type(e).__name__} while processing upload", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": type(e).__name__, "message": str(e), "items": []},
        )

    except Exception as e:
        logger.exception("Unexpected error while processing upload")
        return JSONResponse(
            status_code=500,
            content={"error": "InternalError", "message": str(e), "items": []},
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if not keep_training_assets:
            delete_scan_assets(scan_uuid)


@app.delete("/training-assets/{scan_uuid}", include_in_schema=False)
async def delete_training_assets(
    scan_uuid: uuid.UUID,
    x_cleanup_token: Optional[str] = Header(default=None),
):
    """Remove an unreviewed scan when its owning account is deleted."""
    expected = os.getenv("OCR_CLEANUP_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=503, detail="Asset cleanup is not configured.")
    if not x_cleanup_token or not secrets.compare_digest(x_cleanup_token, expected):
        raise HTTPException(status_code=403, detail="Forbidden.")

    return {
        "deleted": delete_scan_assets(str(scan_uuid)),
        "scan_uuid": str(scan_uuid),
    }
