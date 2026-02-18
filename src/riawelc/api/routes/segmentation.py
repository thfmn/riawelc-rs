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

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from riawelc.api.dependencies import (
    get_seg_augmented_model,
    get_seg_baseline_model,
    get_settings,
    run_inference,
)
from riawelc.api.schemas import UNetSegmentationResponse
from riawelc.inference.predictor import segment_with_unet

router = APIRouter(prefix="/segment/unet", tags=["segmentation"])

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg"}


async def _read_validated_image(file: UploadFile) -> bytes:
    """Read and validate an uploaded image file."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. Must be PNG or JPEG.",
        )
    settings = get_settings()
    image_bytes = await file.read()
    if len(image_bytes) > settings.max_upload_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_upload_size} bytes.",
        )
    return image_bytes


@router.post("/baseline", response_model=UNetSegmentationResponse)
async def segment_baseline(
    file: UploadFile,
    model: object = Depends(get_seg_baseline_model),
) -> UNetSegmentationResponse:
    """Run baseline U-Net segmentation on a welding radiograph."""
    image_bytes = await _read_validated_image(file)
    result = await run_inference(segment_with_unet, model, image_bytes)  # type: ignore[arg-type]
    return UNetSegmentationResponse(**result)


@router.post("/augmented", response_model=UNetSegmentationResponse)
async def segment_augmented(
    file: UploadFile,
    model: object = Depends(get_seg_augmented_model),
) -> UNetSegmentationResponse:
    """Run augmented U-Net segmentation on a welding radiograph."""
    image_bytes = await _read_validated_image(file)
    result = await run_inference(segment_with_unet, model, image_bytes)  # type: ignore[arg-type]
    return UNetSegmentationResponse(**result)
