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

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    class_name: str = Field(description="Predicted defect class name")
    confidence: float = Field(description="Confidence score for the predicted class")
    class_probabilities: dict[str, float] = Field(description="Probability for each class")
    gradcam_base64: str | None = Field(
        default=None, description="Base64-encoded Grad-CAM overlay image (PNG)"
    )


class SegmentationResponse(BaseModel):
    mask_base64: str = Field(description="Base64-encoded segmentation mask (PNG)")
    class_name: str = Field(description="Predicted defect class name")
    confidence: float = Field(description="Confidence score for the predicted class")


class UNetSegmentationResponse(BaseModel):
    mask_base64: str = Field(description="Base64-encoded U-Net segmentation mask (PNG)")
    model_name: str = Field(description="Name of the segmentation model used")


class HealthResponse(BaseModel):
    status: str = Field(description="Service health status")
    version: str = Field(description="API version")


class ModelInfoResponse(BaseModel):
    model_name: str = Field(description="Name of the loaded model")
    input_shape: list[int] = Field(description="Expected input shape [H, W, C]")
    num_classes: int = Field(description="Number of output classes")
    description: str = Field(description="Model description")


class ErrorResponse(BaseModel):
    detail: str = Field(description="Human-readable error description")
    request_id: str | None = Field(default=None, description="Request ID for tracing")


class ReadyResponse(BaseModel):
    status: str = Field(description="Overall readiness status")
    models: dict[str, str] = Field(description="Per-model load status")
