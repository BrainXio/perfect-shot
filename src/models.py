# src/models.py
from pydantic import BaseModel
from typing import List, Optional

class TechnicalQuality(BaseModel):
    sharpness: int = 5
    noise: int = 5
    contrast: int = 5
    exposure: int = 5
    color_balance: int = 5
    saturation: int = 5
    dynamic_range: int = 5
    distortion: int = 5
    artifacts: int = 5

class EnhancementSuggestions(BaseModel):
    crop_bbox: Optional[List[int]] = None
    exposure_factor: Optional[float] = None
    contrast_factor: Optional[float] = None
    sharpen_factor: Optional[float] = None
    auto_enhance: Optional[bool] = None

class ImageAnalysis(BaseModel):
    description: str = ""
    technical_quality: TechnicalQuality = TechnicalQuality()
    lighting_analysis: str = ""
    composition_analysis: str = ""
    detailed_sections: List[str] = []
    metadata: dict = {}
    opencv_metrics: dict = {}