#  Copyright (C) 2026 by Tobias Hoffmann
#  thoffmann-ml@proton.me
#
#  Licensed under the MIT License.
#  For details: https://opensource.org/licenses/MIT
#
#  Author:    Tobias Hoffmann
#  Email:     thoffmann-ml@proton.me
#  License:   MIT
#  Date:      2025-2026
#  Package:   RIAWELC — Welding Defect Classification & Segmentation Pipeline

from __future__ import annotations

import io
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from PIL import Image

from riawelc.api.dependencies import Settings
from riawelc.api.validation import validate_and_read_upload, validate_image_bytes


def _make_png_bytes(width: int = 4, height: int = 4) -> bytes:
    """Create a minimal valid PNG image."""
    img = Image.new("L", (width, height), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_bytes(width: int = 4, height: int = 4) -> bytes:
    """Create a minimal valid JPEG image."""
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# validate_image_bytes
# ---------------------------------------------------------------------------


def test_valid_png_bytes_pass_validation() -> None:
    """PNG magic bytes should be accepted."""
    data = _make_png_bytes()
    result = validate_image_bytes(data)
    assert result is data


def test_valid_jpeg_bytes_pass_validation() -> None:
    """JPEG magic bytes should be accepted."""
    data = _make_jpeg_bytes()
    result = validate_image_bytes(data)
    assert result is data


def test_random_bytes_rejected() -> None:
    """Arbitrary bytes without image magic should raise 415."""
    with pytest.raises(HTTPException) as exc_info:
        validate_image_bytes(b"this is not an image at all")
    assert exc_info.value.status_code == 415


def test_text_file_with_image_content_type_rejected() -> None:
    """Plain text content (no valid magic bytes) must be rejected."""
    with pytest.raises(HTTPException) as exc_info:
        validate_image_bytes(b"Hello, world! I am plain text.")
    assert exc_info.value.status_code == 415


# ---------------------------------------------------------------------------
# validate_and_read_upload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oversized_file_rejected() -> None:
    """File larger than max_upload_size should raise 413."""
    data = _make_png_bytes()
    upload = AsyncMock()
    upload.read = AsyncMock(return_value=data)
    settings = Settings(max_upload_size=10)  # tiny limit

    with pytest.raises(HTTPException) as exc_info:
        await validate_and_read_upload(upload, settings)
    assert exc_info.value.status_code == 413


@pytest.mark.asyncio
async def test_corrupt_image_rejected() -> None:
    """File with valid PNG magic but corrupt body should raise 415."""
    # Start with PNG magic + garbage that PIL cannot decode.
    corrupt_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    upload = AsyncMock()
    upload.read = AsyncMock(return_value=corrupt_data)
    settings = Settings(max_upload_size=10 * 1024 * 1024)

    with pytest.raises(HTTPException) as exc_info:
        await validate_and_read_upload(upload, settings)
    assert exc_info.value.status_code == 415


@pytest.mark.asyncio
async def test_valid_png_upload_accepted() -> None:
    """A valid PNG file within size limits should pass through."""
    data = _make_png_bytes()
    upload = AsyncMock()
    upload.read = AsyncMock(return_value=data)
    settings = Settings(max_upload_size=10 * 1024 * 1024)

    result = await validate_and_read_upload(upload, settings)
    assert result == data


@pytest.mark.asyncio
async def test_valid_jpeg_upload_accepted() -> None:
    """A valid JPEG file within size limits should pass through."""
    data = _make_jpeg_bytes()
    upload = AsyncMock()
    upload.read = AsyncMock(return_value=data)
    settings = Settings(max_upload_size=10 * 1024 * 1024)

    result = await validate_and_read_upload(upload, settings)
    assert result == data
