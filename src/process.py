# src/process.py
from agent import PhotoAgent
from logger import logger
from config import INPUT_DIR, OUTPUT_DIR
from pathlib import Path
import time
import json

agent = PhotoAgent()

input_dir = Path(INPUT_DIR)
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

def is_processed(input_path: Path) -> bool:
    return (output_dir / f"{input_path.stem}_analysis.json").exists()

def analyze_image(input_path: Path):
    logger.info(f"PROCESSING: {input_path.name}")
    analysis = agent.analyze_image(str(input_path))
    logger.info(f"Grade: {analysis.get('grade', 'N/A')}/10")

    json_path = output_dir / f"{input_path.stem}_analysis.json"
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"SAVED: {json_path.name}")

def main():
    logger.info(f"Monitoring {input_dir}")
    while True:
        found = False
        for img_path in input_dir.rglob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"} and not img_path.name.startswith(".") and not is_processed(img_path):
                found = True
                analyze_image(img_path)
        if not found:
            logger.info("No unprocessed images")
        time.sleep(10)

if __name__ == "__main__":
    main()