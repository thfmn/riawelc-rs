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
from typing import TYPE_CHECKING

from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

if TYPE_CHECKING:
    from riawelc.api.dependencies import Settings

# Magic byte prefixes for supported image formats.
_PNG_MAGIC = b"\x89PNG"
_JPEG_MAGIC = b"\xff\xd8\xff"


def validate_image_bytes(data: bytes) -> bytes:
    """Validate that *data* begins with PNG or JPEG magic bytes.

    Returns the data unchanged if valid; raises ``HTTPException(415)``
    otherwise.
    """
    if data[:4] == _PNG_MAGIC or data[:3] == _JPEG_MAGIC:
        return data
    raise HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail="Invalid image file. Only PNG and JPEG files are accepted.",
    )


async def validate_and_read_upload(file: UploadFile, settings: Settings) -> bytes:
    """Read an uploaded file and validate size, magic bytes, and PIL decodability.

    Raises:
        HTTPException(413): File exceeds ``settings.max_upload_size``.
        HTTPException(415): File is not a valid PNG/JPEG image.
    """
    image_bytes = await file.read()

    if len(image_bytes) > settings.max_upload_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_upload_size} bytes.",
        )

    validate_image_bytes(image_bytes)

    # Verify that PIL can actually decode the image.
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
    except (UnidentifiedImageError, SyntaxError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Corrupt or unreadable image file: {exc}",
        ) from exc

    return image_bytes
