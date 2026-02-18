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

from riawelc.api.dependencies import get_model, get_settings, run_inference
from riawelc.api.schemas import ModelInfoResponse, PredictionResponse, SegmentationResponse
from riawelc.api.validation import validate_and_read_upload
from riawelc.data import CLASS_NAMES
from riawelc.inference.predictor import WeldingDefectPredictor

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile,
    model: object = Depends(get_model),
) -> PredictionResponse:
    """Classify a welding radiograph and return prediction with Grad-CAM overlay."""
    image_bytes = await validate_and_read_upload(file, get_settings())
    predictor = WeldingDefectPredictor(model)  # type: ignore[arg-type]
    result = await run_inference(predictor.predict, image_bytes)
    return PredictionResponse(**result)


@router.post("/segment", response_model=SegmentationResponse)
async def segment(
    file: UploadFile,
    model: object = Depends(get_model),
) -> SegmentationResponse:
    """Generate a weakly-supervised segmentation mask for a welding radiograph."""
    image_bytes = await validate_and_read_upload(file, get_settings())
    predictor = WeldingDefectPredictor(model)  # type: ignore[arg-type]
    result = await run_inference(predictor.segment, image_bytes)
    return SegmentationResponse(**result)


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info(model: object = Depends(get_model)) -> ModelInfoResponse:
    """Return metadata about the loaded classification model."""
    input_shape = list(model.input_shape[1:])  # type: ignore[union-attr]
    return ModelInfoResponse(
        model_name=model.name,  # type: ignore[union-attr]
        input_shape=input_shape,
        num_classes=len(CLASS_NAMES),
        description="Welding defect classification model (EfficientNetB0 backbone)",
    )
