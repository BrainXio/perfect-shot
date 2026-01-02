# src/process.py
from agent import PhotoAgent
from tools import crop, adjust_exposure, adjust_contrast, sharpen, auto_enhance, compare, add_watermark
from PIL import Image
import sys
from pathlib import Path

agent = PhotoAgent()

TOOL_MAP = {
    "crop": lambda current, args: crop(current, tuple(args["bbox"])),
    "adjust_exposure": lambda current, args: adjust_exposure(current, args["factor"]),
    "adjust_contrast": lambda current, args: adjust_contrast(current, args["factor"]),
    "sharpen": lambda current, args: sharpen(current, args["factor"]),
    "auto_enhance": lambda current, _: auto_enhance(current),
}

def process_single(input_path: Path):
    original = Image.open(input_path)
    current = original.copy()
    
    tools = agent.analyze_and_plan(str(input_path))
    print(f"{input_path.name}: {tools}")
    
    for tool in tools:
        name = tool.get("name")
        args = tool.get("args", {})
        if name in TOOL_MAP:
            try:
                current = TOOL_MAP[name](current, args)
            except Exception as e:
                print(f"Error applying {name} to {input_path}: {e}")
    
    current = add_watermark(current)
    
    output_path = input_path.parent / f"{input_path.stem}_enhanced{input_path.suffix}"
    compare_path = input_path.parent / f"{input_path.stem}_enhanced_compare{input_path.suffix}"
    
    current.save(output_path)
    compare(original, current).save(compare_path)
    
    print(f"Saved: {output_path.name} and {compare_path.name}")

def process_batch(directory: str):
    dir_path = Path(directory)
    if not dir_path.is_dir():
        print(f"Error: {directory} is not a directory")
        sys.exit(1)
    
    images = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.jpeg")) + list(dir_path.glob("*.png"))
    if not images:
        print("No supported images found")
        return
    
    print(f"Processing {len(images)} images in {dir_path}")
    for img_path in images:
        try:
            process_single(img_path)
        except Exception as e:
            print(f"Failed on {img_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.process <image_file_or_directory>")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    if path.is_file():
        process_single(path)
    elif path.is_dir():
        process_batch(str(path))
    else:
        print("Error: Path not found")
        sys.exit(1)