# Updated src/process.py (robust tool dispatch)
from agent import PhotoAgent
from tools import *
from PIL import Image

agent = PhotoAgent()

TOOL_MAP = {
    "crop": lambda args: crop(current, tuple(args["bbox"])),
    "adjust_exposure": lambda args: adjust_exposure(current, args["factor"]),
    "adjust_contrast": lambda args: adjust_contrast(current, args["factor"]),
    "sharpen": lambda args: sharpen(current, args["factor"]),
    "auto_enhance": lambda _: auto_enhance(current),
}

def process_image(input_path: str, output_path: str):
    global current
    original = Image.open(input_path)
    current = original.copy()
    
    tools = agent.analyze_and_plan(input_path)
    print(f"Planned tools: {tools}")
    
    for tool in tools:
        name = tool.get("name")
        args = tool.get("args", {})
        if name in TOOL_MAP:
            current = TOOL_MAP[name](args)
    
    current.save(output_path)
    compare(original, current).save(output_path.replace(".jpg", "_compare.jpg"))

if __name__ == "__main__":
    import sys
    process_image(sys.argv[1], sys.argv[2])