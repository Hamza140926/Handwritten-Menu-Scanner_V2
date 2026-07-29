"""Best-effort Cloudinary storage for OCR training assets.

The OCR service must keep scanning when storage is unavailable, so every
operation returns ``None`` on failure instead of raising into the pipeline.
Credentials are read from the process environment:

    CLOUDINARY_CLOUD_NAME
    CLOUDINARY_API_KEY
    CLOUDINARY_API_SECRET

Set CLOUDINARY_UPLOAD_ENABLED=0 to explicitly disable uploads.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Any

from logging_config import get_logger

logger = get_logger(__name__)

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    # Environment variables still work without python-dotenv.
    pass


def is_configured() -> bool:
    if os.getenv("CLOUDINARY_UPLOAD_ENABLED", "1").strip().lower() in {
        "0", "false", "no", "off",
    }:
        return False
    if os.getenv("CLOUDINARY_URL"):
        return True
    return all(
        os.getenv(name)
        for name in (
            "CLOUDINARY_CLOUD_NAME",
            "CLOUDINARY_API_KEY",
            "CLOUDINARY_API_SECRET",
        )
    )


def _cloudinary_modules():
    import cloudinary
    import cloudinary.uploader

    if os.getenv("CLOUDINARY_URL"):
        # The SDK reads cloudinary://<key>:<secret>@<cloud-name>.
        cloudinary.config(secure=True)
    else:
        cloudinary.config(
            cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
            api_key=os.environ["CLOUDINARY_API_KEY"],
            api_secret=os.environ["CLOUDINARY_API_SECRET"],
            secure=True,
        )
    return cloudinary.uploader


def _cloudinary_api():
    import cloudinary
    import cloudinary.api

    if os.getenv("CLOUDINARY_URL"):
        cloudinary.config(secure=True)
    else:
        cloudinary.config(
            cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
            api_key=os.environ["CLOUDINARY_API_KEY"],
            api_secret=os.environ["CLOUDINARY_API_SECRET"],
            secure=True,
        )
    return cloudinary.api


def upload_image(
    source: str | Path | bytes | bytearray,
    public_id: str,
    *,
    retries: int = 2,
) -> dict[str, Any] | None:
    """Upload an image and return stable asset metadata.

    ``public_id`` is relative to the ``scantosee`` folder. Raw byte input is
    wrapped in BytesIO because the Cloudinary SDK expects a path or file-like
    object. Upload failures are intentionally non-fatal.
    """
    if not is_configured():
        logger.debug("Cloudinary upload skipped: credentials are not configured")
        return None

    try:
        uploader = _cloudinary_modules()
    except Exception:
        logger.exception("Cloudinary SDK/configuration is unavailable")
        return None

    upload_source = (
        io.BytesIO(bytes(source))
        if isinstance(source, (bytes, bytearray))
        else str(source)
    )

    for attempt in range(1, max(1, retries) + 1):
        try:
            if hasattr(upload_source, "seek"):
                upload_source.seek(0)
            result = uploader.upload(
                upload_source,
                public_id=public_id,
                folder="scantosee",
                overwrite=True,
                invalidate=False,
                resource_type="image",
                tags=["scantosee", "ocr-training"],
            )
            return {
                "url": result.get("secure_url"),
                "public_id": result.get("public_id"),
                "asset_id": result.get("asset_id"),
                "version": result.get("version"),
                "bytes": result.get("bytes"),
                "format": result.get("format"),
            }
        except Exception as exc:
            logger.warning(
                "Cloudinary upload failed",
                extra={
                    "public_id": public_id,
                    "attempt": attempt,
                    "max_attempts": retries,
                    "error": str(exc),
                },
            )
            if attempt < retries:
                time.sleep(0.25 * attempt)

    logger.error("Cloudinary upload abandoned", extra={"public_id": public_id})
    return None


def delete_scan_assets(scan_uuid: str) -> bool:
    """Best-effort removal of every Cloudinary image for one scan."""
    if not scan_uuid or not is_configured():
        return False

    try:
        api = _cloudinary_api()
        api.delete_resources_by_prefix(
            f"scantosee/menus/{scan_uuid}/",
            resource_type="image",
            invalidate=True,
        )
        logger.info("Cloudinary scan assets deleted", extra={"scan_uuid": scan_uuid})
        return True
    except Exception:
        logger.exception(
            "Cloudinary scan asset cleanup failed",
            extra={"scan_uuid": scan_uuid},
        )
        return False
