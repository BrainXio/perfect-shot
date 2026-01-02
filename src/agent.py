# src/agent.py
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from PIL import Image
from config import (
    MODEL_NAME, QUANTIZATION, MAX_MODEL_LEN,
    GPU_MEMORY_UTILIZATION, TEMPERATURE, MAX_TOKENS
)
import json

class PhotoAgent:
    def __init__(self):
        self.llm = LLM(
            model=MODEL_NAME,
            quantization=QUANTIZATION,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        )
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME, min_pixels=256*28*28, max_pixels=1280*28*28)

    def analyze_and_plan(self, image_path: str) -> list[dict]:
        prompt = [
            {"role": "system", "content": "You are a photo enhancement expert. Output ONLY a JSON array of tool calls."},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": (
                    "Analyze this photo and suggest improvements using tools: "
                    "crop(bbox=[left,top,right,bottom]), adjust_exposure(factor=float), "
                    "adjust_contrast(factor=float), sharpen(factor=float), auto_enhance(). "
                    "Return ONLY JSON array of tool calls."
                )}
            ]}
        ]

        inputs = self.processor(prompt, images=[Image.open(image_path)], return_tensors="pt")
        sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

        outputs = self.llm.generate(inputs, sampling_params)

        try:
            text = outputs[0].outputs[0].text.strip()
            return json.loads(text)
        except Exception as e:
            print(f"JSON parse error: {e}\nRaw output: {text}")
            return []