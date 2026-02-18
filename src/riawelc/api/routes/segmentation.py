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

from fastapi import APIRouter, Depends, UploadFile

from riawelc.api.dependencies import (
    get_seg_augmented_model,
    get_seg_baseline_model,
    get_settings,
    run_inference,
)
from riawelc.api.schemas import UNetSegmentationResponse
from riawelc.api.validation import validate_and_read_upload
from riawelc.inference.predictor import segment_with_unet

router = APIRouter(prefix="/segment/unet", tags=["segmentation"])


@router.post("/baseline", response_model=UNetSegmentationResponse)
async def segment_baseline(
    file: UploadFile,
    model: object = Depends(get_seg_baseline_model),
) -> UNetSegmentationResponse:
    """Run baseline U-Net segmentation on a welding radiograph."""
    image_bytes = await validate_and_read_upload(file, get_settings())
    result = await run_inference(segment_with_unet, model, image_bytes)  # type: ignore[arg-type]
    return UNetSegmentationResponse(**result)


@router.post("/augmented", response_model=UNetSegmentationResponse)
async def segment_augmented(
    file: UploadFile,
    model: object = Depends(get_seg_augmented_model),
) -> UNetSegmentationResponse:
    """Run augmented U-Net segmentation on a welding radiograph."""
    image_bytes = await validate_and_read_upload(file, get_settings())
    result = await run_inference(segment_with_unet, model, image_bytes)  # type: ignore[arg-type]
    return UNetSegmentationResponse(**result)
