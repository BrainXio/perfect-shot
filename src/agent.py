# Updated src/agent.py (fix inputs for vLLM + processor)
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from PIL import Image
import json
import base64
import io

class PhotoAgent:
    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct-AWQ"):
        self.llm = LLM(model=model_name, quantization="awq", max_model_len=4096)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def analyze_and_plan(self, image_path: str) -> list[dict]:
        image_b64 = self._encode_image(image_path)
        
        prompt = [
            {"role": "system", "content": "You are a photo enhancement expert. Output ONLY a JSON array of tool calls."},
            {"role": "user", "content": [
                {"type": "image", "image": f"data:image/jpeg;base64,{image_b64}"},
                {"type": "text", "text": (
                    "Analyze this photo and suggest improvements. Available tools: "
                    "crop(bbox=[left,top,right,bottom]), adjust_exposure(factor=float), "
                    "adjust_contrast(factor=float), sharpen(factor=float), auto_enhance(). "
                    "Return ONLY JSON array like: [{\"name\": \"crop\", \"args\": {\"bbox\": [0,0,100,100]}}, ...]"
                )}
            ]}
        ]
        
        sampling_params = SamplingParams(temperature=0.3, max_tokens=512, stop=["\n\n"])
        outputs = self.llm.generate(prompt, sampling_params, use_tqdm=False)
        
        try:
            text = outputs[0].outputs[0].text.strip()
            return json.loads(text)
        except Exception as e:
            print(f"Parse error: {e}\nRaw: {text}")
            return []