"""Pydantic request/response models."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: Optional[str] = Field(default=None, max_length=2000)
    width: int = Field(default=1024, ge=256, le=4096)
    height: int = Field(default=1024, ge=256, le=4096)
    steps: int = Field(default=28, ge=1, le=200)
    guidance: float = Field(default=3.0, ge=0.0, le=20.0)
    seed: Optional[int] = Field(default=None, ge=0, le=2**32 - 1)
    num_images: int = Field(default=1, ge=1, le=8)


class JobImage(BaseModel):
    url: str
    seed: int
    filename: str


class JobStatus(BaseModel):
    id: str
    state: str  # queued | loading_model | running | done | error
    progress: float  # 0.0 – 1.0
    step: int
    total_steps: int
    prompt: str
    images: List[JobImage] = []
    error: Optional[str] = None
    created_at: float
    elapsed: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    dtype: str
    base_model: str
    gguf: str
    gguf_present: bool
