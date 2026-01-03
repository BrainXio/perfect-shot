# src/agent.py
import ollama
from logger import logger
from config import VISION_MODEL, AGENT_LANGUAGE
from PIL import Image
from pathlib import Path
import tempfile
from models import ImageAnalysis, TechnicalQuality, EnhancementSuggestions
import json
import re
from tools import get_opencv_metrics, crop, adjust_exposure, adjust_contrast, sharpen, auto_enhance, add_watermark

class PhotoAgent:
    def __init__(self):
        logger.info("Initializing PhotoAgent...")
        try:
            ollama.show(VISION_MODEL)
            logger.info(f"Model {VISION_MODEL} available")
        except:
            logger.info(f"Pulling {VISION_MODEL}...")
            ollama.pull(VISION_MODEL)
            logger.info("Model pulled")
        
        logger.info("Ready")

    def _generate(self, prompt: str, image_path: str) -> str:
        with open(image_path, "rb") as f:
            img = f.read()
        response = ollama.generate(
            model=VISION_MODEL,
            prompt=prompt,
            images=[img]
        )
        return response["response"].strip()

    def describe_image(self, image_path: str) -> str:
        logger.info("Step 1: Describing image")
        prompt = f"Respond in {AGENT_LANGUAGE}. Describe the image in detail as a professional photographer. One clear paragraph."
        return self._generate(prompt, image_path)

    def analyze_technical_quality(self, image_path: str) -> TechnicalQuality:
        logger.info("Step 2: Analyzing technical quality")
        prompt = f"""Respond in {AGENT_LANGUAGE}. Rate each technical aspect from 1 to 10 (10 = excellent/perfect, 1 = poor). 
For noise, distortion, and artifacts: 10 means minimal or none.
Output ONLY valid JSON with keys: sharpness, noise, contrast, exposure, color_balance, saturation, dynamic_range, distortion, artifacts."""
        text = self._generate(prompt, image_path)
        logger.info(f"Raw technical response: {text}")
        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL) or re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        try:
            data = json.loads(text)
            validated = {}
            for key in TechnicalQuality.model_fields:
                val = data.get(key, 5)
                validated[key] = int(val) if isinstance(val, (int, float)) and 1 <= val <= 10 else 5
            return TechnicalQuality(**validated)
        except Exception as e:
            logger.error(f"Technical parse failed: {e}")
            return TechnicalQuality()

    def analyze_lighting(self, image_path: str) -> str:
        logger.info("Step 3: Analyzing lighting")
        prompt = f"Respond in {AGENT_LANGUAGE}. Analyze lighting: direction, quality (soft/hard), temperature, shadows, highlights, mood created by light. One paragraph."
        return self._generate(prompt, image_path)

    def analyze_composition(self, image_path: str) -> str:
        logger.info("Step 4: Analyzing composition")
        prompt = f"Respond in {AGENT_LANGUAGE}. Analyze composition: rule of thirds, leading lines, symmetry, framing, balance, focal point, negative space. One paragraph."
        return self._generate(prompt, image_path)

    def analyze_image_details(self, image_path: str) -> list[str]:
        logger.info("Step 5: Analyzing detailed sections")
        img = Image.open(image_path)
        w, h = img.size
        grid_size = 3 if max(w, h) > 1000 else 2
        details = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(grid_size):
                for j in range(grid_size):
                    left = j * w // grid_size
                    top = i * h // grid_size
                    right = (j + 1) * w // grid_size
                    bottom = (i + 1) * h // grid_size
                    crop_img = img.crop((left, top, right, bottom))
                    crop_path = Path(tmpdir) / f"grid_{i}_{j}.jpg"
                    crop_img.save(crop_path)
                    prompt = f"Respond in {AGENT_LANGUAGE}. Describe this section in intricate detail. One sentence."
                    detail = self._generate(prompt, str(crop_path))
                    details.append(detail)
        return details

    def extract_metadata(self, image_path: str) -> dict:
        try:
            img = Image.open(image_path)
            exif = img.getexif()
            return {k: str(v) for k, v in exif.items()} if exif else {}
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return {}

    def _calculate_grade(self, technical_quality: TechnicalQuality) -> float:
        scores = [getattr(technical_quality, field) for field in TechnicalQuality.model_fields]
        return round(sum(scores) / len(scores), 1)

    def analyze_image(self, image_path: str) -> dict:
        description = self.describe_image(image_path)
        technical = self.analyze_technical_quality(image_path)
        lighting = self.analyze_lighting(image_path)
        composition = self.analyze_composition(image_path)
        details = self.analyze_image_details(image_path)
        metadata = self.extract_metadata(image_path)
        opencv_metrics = get_opencv_metrics(image_path)
        
        analysis = ImageAnalysis(
            description=description,
            technical_quality=technical,
            lighting_analysis=lighting,
            composition_analysis=composition,
            detailed_sections=details,
            metadata=metadata,
            opencv_metrics=opencv_metrics
        )
        analysis_dict = analysis.model_dump()
        analysis_dict["grade"] = self._calculate_grade(technical)
        return analysis_dict

    def get_enhancement_suggestions(self, analysis_dict: dict, image_path: str) -> EnhancementSuggestions:
        technical = analysis_dict["technical_quality"]
        tech_scores = "\n".join(f"{k.replace('_', ' ').capitalize()}: {v}/10" for k, v in technical.items())
        
        base_prompt = f"""You are a professional photo editor specializing in subtle, natural enhancements. 
Preserve the original intent, style, and atmosphere of the photograph. 
Make only minimal, conservative tweaks that gently improve flaws without altering the image's character.

Overall grade: {analysis_dict["grade"]}/10
Technical ratings:
{tech_scores}

Description: {analysis_dict["description"]}
Lighting: {analysis_dict["lighting_analysis"]}
Composition: {analysis_dict["composition_analysis"]}

Rules:
- Only suggest changes for aspects rated below 8/10.
- Prefer NO change if improvement would be marginal.
- Be extremely conservative: small tweaks only.
- exposure_factor / contrast_factor: 0.95–1.05 only (null otherwise)
- sharpen_factor: 0.0–0.8 only (null otherwise)
- auto_enhance: true only if shadows/highlights are clearly clipped
- crop_bbox: only if composition is significantly flawed AND cropping to rule of thirds or removing clear distractions greatly helps; keep as much original content as possible; null otherwise

Output ONLY a single flat JSON object. No markdown, no ```json, no extra keys.

Allowed keys (use null if not suggesting):
- "crop_bbox": [left, top, right, bottom] (4 integers) or null
- "exposure_factor": float or null
- "contrast_factor": float or null
- "sharpen_factor": float or null
- "auto_enhance": true/false/null"""

        allowed_keys = set(EnhancementSuggestions.model_fields.keys())

        for attempt in range(3):
            response = self._generate(base_prompt if attempt == 0 else feedback_prompt, image_path)
            logger.info(f"Raw suggestions attempt {attempt+1}: {response}")

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            json_text = json_match.group(0) if json_match else response

            try:
                data = json.loads(json_text)
                if not isinstance(data, dict):
                    raise ValueError("Not a JSON object")
                invalid_keys = set(data.keys()) - allowed_keys
                if invalid_keys:
                    raise ValueError(f"Invalid keys: {invalid_keys}")
                cleaned = {k: data.get(k) for k in allowed_keys}
                return EnhancementSuggestions(**cleaned)
            except Exception as e:
                logger.warning(f"Suggestions parse failed attempt {attempt+1}: {e}")
                if attempt == 2:
                    return EnhancementSuggestions()
                feedback_prompt = f"""{base_prompt}

Your previous output was invalid.
Previous: {response}
Error: {str(e)}

Fix it and output ONLY the correct flat JSON."""

    def apply_enhancements(self, image_path: str, suggestions: EnhancementSuggestions) -> Image.Image:
        img = Image.open(image_path).convert("RGB")
        if suggestions.crop_bbox:
            try:
                bbox = tuple(map(int, suggestions.crop_bbox))
                if all(x >= 0 for x in bbox[:2]) and bbox[2] > bbox[0] and bbox[3] > bbox[1] and bbox[2] <= img.width and bbox[3] <= img.height:
                    img = crop(img, bbox)
                    logger.info(f"Applied crop: {bbox}")
            except Exception:
                logger.error("Invalid crop bbox")
        if suggestions.auto_enhance is True:
            img = auto_enhance(img)
            logger.info("Applied auto_enhance")
        if suggestions.exposure_factor is not None and suggestions.exposure_factor != 1.0:
            img = adjust_exposure(img, suggestions.exposure_factor)
            logger.info(f"Applied exposure: {suggestions.exposure_factor}")
        if suggestions.contrast_factor is not None and suggestions.contrast_factor != 1.0:
            img = adjust_contrast(img, suggestions.contrast_factor)
            logger.info(f"Applied contrast: {suggestions.contrast_factor}")
        if suggestions.sharpen_factor is not None and suggestions.sharpen_factor > 0:
            img = sharpen(img, suggestions.sharpen_factor)
            logger.info(f"Applied sharpen: {suggestions.sharpen_factor}")
        return img