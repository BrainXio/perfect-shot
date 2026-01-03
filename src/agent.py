# src/agent.py
import ollama, re, json, tempfile
from logger import logger
from config import VISION_MODEL, AGENT_LANGUAGE
from PIL import Image
from pathlib import Path
from models import ImageAnalysis, TechnicalQuality, EnhancementSuggestions
from tools import get_opencv_metrics, crop, adjust_exposure, adjust_contrast, sharpen, auto_enhance

class PhotoAgent:
    def __init__(self):
        logger.info("Initializing PhotoAgent...")
        try:
            ollama.show(VISION_MODEL)
        except:
            ollama.pull(VISION_MODEL)
        logger.info("Ready")

    def _generate(self, prompt: str, image_path: str) -> str:
        with open(image_path, "rb") as f:
            img = f.read()
        return ollama.generate(model=VISION_MODEL, prompt=prompt, images=[img])["response"].strip()

    def describe_image(self, image_path: str) -> str:
        return self._generate(f"Respond in {AGENT_LANGUAGE}. Describe the image in detail as a professional photographer. One clear paragraph.", image_path)

    def analyze_technical_quality(self, image_path: str) -> TechnicalQuality:
        text = self._generate(f"""Respond in {AGENT_LANGUAGE}. Rate each aspect 1-10 (10=excellent). For noise/distortion/artifacts: 10=minimal.
Output ONLY JSON with keys: sharpness, noise, contrast, exposure, color_balance, saturation, dynamic_range, distortion, artifacts.""", image_path)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(match.group(0) if match else text)
        validated = {k: int(data.get(k, 5)) if 1 <= data.get(k, 5) <= 10 else 5 for k in TechnicalQuality.model_fields}
        return TechnicalQuality(**validated)

    def analyze_lighting(self, image_path: str) -> str:
        return self._generate(f"Respond in {AGENT_LANGUAGE}. Analyze lighting: direction, quality, temperature, shadows, highlights, mood. One paragraph.", image_path)

    def analyze_composition(self, image_path: str) -> str:
        return self._generate(f"Respond in {AGENT_LANGUAGE}. Analyze composition: rule of thirds, leading lines, symmetry, framing, balance, focal point, negative space. One paragraph.", image_path)

    def analyze_image_details(self, image_path: str) -> list[str]:
        img = Image.open(image_path)
        grid_size = 3 if max(img.size) > 1000 else 2
        details = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(grid_size):
                for j in range(grid_size):
                    box = (j*img.width//grid_size, i*img.height//grid_size, (j+1)*img.width//grid_size, (i+1)*img.height//grid_size)
                    crop_path = Path(tmpdir) / f"grid_{i}_{j}.jpg"
                    img.crop(box).save(crop_path)
                    details.append(self._generate(f"Respond in {AGENT_LANGUAGE}. Describe this section in intricate detail. One sentence.", str(crop_path)))
        return details

    def extract_metadata(self, image_path: str) -> dict:
        try:
            return {k: str(v) for k, v in Image.open(image_path).getexif().items()} if Image.open(image_path).getexif() else {}
        except:
            return {}

    def _calculate_grade(self, tech: TechnicalQuality) -> float:
        return round(sum(getattr(tech, f) for f in TechnicalQuality.model_fields) / len(TechnicalQuality.model_fields), 1)

    def analyze_image(self, image_path: str) -> dict:
        analysis = ImageAnalysis(
            description=self.describe_image(image_path),
            technical_quality=self.analyze_technical_quality(image_path),
            lighting_analysis=self.analyze_lighting(image_path),
            composition_analysis=self.analyze_composition(image_path),
            detailed_sections=self.analyze_image_details(image_path),
            metadata=self.extract_metadata(image_path),
            opencv_metrics=get_opencv_metrics(image_path)
        )
        d = analysis.model_dump()
        d["grade"] = self._calculate_grade(analysis.technical_quality)
        return d

    def get_enhancement_suggestions(self, analysis_dict: dict, image_path: str, prev_grade: float = 0.0) -> EnhancementSuggestions:
        tech_scores = "\n".join(f"{k.replace('_', ' ').capitalize()}: {v}/10" for k, v in analysis_dict["technical_quality"].items())
        prompt = f"""Meticulous editor: gradual improvements only.
Current grade: {analysis_dict["grade"]}/10 (prev: {prev_grade}/10)
Make 1-2 small changes max if needed. Stop if >=9.0.

Ratings:
{tech_scores}

Description: {analysis_dict["description"]}

Adjustments (null=no change):
- crop_bbox: [l,t,r,b]
- exposure_factor: 0.95–1.05
- contrast_factor: 0.95–1.05
- sharpen_factor: 0.0–0.6
- auto_enhance: true/false
- vignette_strength: 0.0–0.3
- color_boost: 1.0–1.08

Output ONLY flat JSON."""
        for _ in range(3):
            text = self._generate(prompt, image_path)
            match = re.search(r"\{.*\}", text, re.DOTALL)
            try:
                data = json.loads(match.group(0) if match else text)
                return EnhancementSuggestions(**{k: data.get(k) for k in EnhancementSuggestions.model_fields})
            except:
                pass
        return EnhancementSuggestions()

    def apply_enhancements(self, image_path: str, suggestions: EnhancementSuggestions) -> Image.Image:
        img = Image.open(image_path).convert("RGB")
        if suggestions.crop_bbox:
            img = crop(img, tuple(map(int, suggestions.crop_bbox)))
        if suggestions.auto_enhance:
            img = auto_enhance(img)
        if suggestions.exposure_factor is not None:
            img = adjust_exposure(img, suggestions.exposure_factor)
        if suggestions.contrast_factor is not None:
            img = adjust_contrast(img, suggestions.contrast_factor)
        if suggestions.sharpen_factor is not None:
            img = sharpen(img, suggestions.sharpen_factor)
        return img

    def generate_caption(self, image_path: str) -> str:
        return self._generate(f"Respond in {AGENT_LANGUAGE}. Short witty Grok quote (max 12 words) on curiosity or beauty.", image_path)