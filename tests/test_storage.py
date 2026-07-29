"""Tests for best-effort Cloudinary training-asset storage."""

import storage


def test_unconfigured_storage_is_non_fatal(monkeypatch):
    for name in (
        "CLOUDINARY_URL",
        "CLOUDINARY_CLOUD_NAME",
        "CLOUDINARY_API_KEY",
        "CLOUDINARY_API_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)

    assert storage.is_configured() is False
    assert storage.upload_image(b"image", "menus/test/region") is None


def test_upload_returns_stable_asset_metadata(monkeypatch):
    monkeypatch.setenv("CLOUDINARY_CLOUD_NAME", "test")
    monkeypatch.setenv("CLOUDINARY_API_KEY", "key")
    monkeypatch.setenv("CLOUDINARY_API_SECRET", "secret")

    class FakeUploader:
        @staticmethod
        def upload(source, **kwargs):
            assert source.read() == b"jpeg"
            assert kwargs["public_id"] == "menus/scan/regions/0001"
            return {
                "secure_url": "https://example.test/crop.jpg",
                "public_id": "scantosee/menus/scan/regions/0001",
                "asset_id": "asset-1",
                "version": 7,
                "bytes": 4,
                "format": "jpg",
            }

    monkeypatch.setattr(storage, "_cloudinary_modules", lambda: FakeUploader)
    result = storage.upload_image(b"jpeg", "menus/scan/regions/0001")

    assert result == {
        "url": "https://example.test/crop.jpg",
        "public_id": "scantosee/menus/scan/regions/0001",
        "asset_id": "asset-1",
        "version": 7,
        "bytes": 4,
        "format": "jpg",
    }


def test_upload_failure_is_non_fatal(monkeypatch):
    monkeypatch.setenv("CLOUDINARY_CLOUD_NAME", "test")
    monkeypatch.setenv("CLOUDINARY_API_KEY", "key")
    monkeypatch.setenv("CLOUDINARY_API_SECRET", "secret")

    class FailingUploader:
        @staticmethod
        def upload(source, **kwargs):
            raise OSError("network down")

    monkeypatch.setattr(storage, "_cloudinary_modules", lambda: FailingUploader)
    assert storage.upload_image(b"jpeg", "menus/scan/regions/0001", retries=1) is None


def test_delete_scan_assets_uses_scoped_prefix(monkeypatch):
    monkeypatch.setenv("CLOUDINARY_CLOUD_NAME", "test")
    monkeypatch.setenv("CLOUDINARY_API_KEY", "key")
    monkeypatch.setenv("CLOUDINARY_API_SECRET", "secret")
    calls = []

    class FakeApi:
        @staticmethod
        def delete_resources_by_prefix(prefix, **kwargs):
            calls.append((prefix, kwargs))

    monkeypatch.setattr(storage, "_cloudinary_api", lambda: FakeApi)

    assert storage.delete_scan_assets("scan-123") is True
    assert calls == [(
        "scantosee/menus/scan-123/",
        {"resource_type": "image", "invalidate": True},
    )]


def test_delete_scan_assets_failure_is_non_fatal(monkeypatch):
    monkeypatch.setenv("CLOUDINARY_CLOUD_NAME", "test")
    monkeypatch.setenv("CLOUDINARY_API_KEY", "key")
    monkeypatch.setenv("CLOUDINARY_API_SECRET", "secret")

    class FailingApi:
        @staticmethod
        def delete_resources_by_prefix(prefix, **kwargs):
            raise OSError("network down")

    monkeypatch.setattr(storage, "_cloudinary_api", lambda: FailingApi)
    assert storage.delete_scan_assets("scan-123") is False
