# src/models.py
from pydantic import BaseModel
from typing import List

class TechnicalQuality(BaseModel):
    sharpness: str = ""
    noise: str = ""
    contrast: str = ""
    exposure: str = ""
    color_balance: str = ""
    saturation: str = ""
    dynamic_range: str = ""
    distortion: str = ""
    artifacts: str = ""

class ImageAnalysis(BaseModel):
    description: str = ""
    technical_quality: TechnicalQuality = TechnicalQuality()
    lighting_analysis: str = ""
    composition_analysis: str = ""
    detailed_sections: List[str] = []
    metadata: dict = {}
    opencv_metrics: dict = {}