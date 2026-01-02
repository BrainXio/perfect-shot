# src/agent.py
import ollama
from logger import logger
from config import VISION_MODEL, AGENT_LANGUAGE
from PIL import Image
from pathlib import Path
import tempfile
from models import ImageAnalysis, TechnicalQuality
import json
import re
from tools import get_opencv_metrics

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
        prompt = f"Respond in {AGENT_LANGUAGE}. Rate each technical aspect from 1 to 10. Output ONLY valid JSON with keys: sharpness, noise, contrast, exposure, color_balance, saturation, dynamic_range, distortion, artifacts."
        text = self._generate(prompt, image_path)
        logger.info(f"Raw technical response: {text}")
        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        try:
            data = json.loads(text)
            return TechnicalQuality(
                sharpness=str(data.get("sharpness", 5)),
                noise=str(data.get("noise", 5)),
                contrast=str(data.get("contrast", 5)),
                exposure=str(data.get("exposure", 5)),
                color_balance=str(data.get("color_balance", 5)),
                saturation=str(data.get("saturation", 5)),
                dynamic_range=str(data.get("dynamic_range", 5)),
                distortion=str(data.get("distortion", 5)),
                artifacts=str(data.get("artifacts", 5))
            )
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
        return analysis.model_dump()