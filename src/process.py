# src/process.py
from agent import PhotoAgent
from logger import logger
from config import INPUT_DIR, OUTPUT_DIR, CACHE_DIR
from pathlib import Path
from PIL import Image
import time, mlflow
from tools import compare, add_watermark, professional_polish
from models import EnhancementSuggestions

mlflow.set_tracking_uri("file:///mlruns")  # simplified

agent = PhotoAgent()
input_dir, output_dir, cache_dir = map(Path, [INPUT_DIR, OUTPUT_DIR, CACHE_DIR])
for p in [output_dir, cache_dir]:
    p.mkdir(parents=True, exist_ok=True)

def is_processed(p: Path) -> bool:
    return (output_dir / f"{p.stem}_perfect.jpg").exists()

def process_photo(input_path: Path):
    stem = input_path.stem
    current_path = str(input_path)
    prev_grade = 0.0
    iteration = 0
    last_suggestions = EnhancementSuggestions()

    with mlflow.start_run(run_name=f"{stem}_{time.strftime('%Y%m%d-%H%M%S')}"):
        while iteration < 6:
            analysis = agent.analyze_image(current_path)
            grade = analysis["grade"]
            logger.info(f"Iter {iteration} grade: {grade}")

            if grade >= 9.2 or (grade - prev_grade < 0.2 and iteration > 1):
                break

            suggestions = agent.get_enhancement_suggestions(analysis, current_path, prev_grade)
            last_suggestions = suggestions
            if not any(getattr(suggestions, f) for f in suggestions.model_fields if getattr(suggestions, f) is not None):
                break

            enhanced = agent.apply_enhancements(current_path, suggestions)
            enhanced_path = cache_dir / f"{stem}_v{iteration}.jpg"
            enhanced.save(enhanced_path)
            current_path = str(enhanced_path)
            prev_grade = grade
            iteration += 1

        final = Image.open(current_path)
        caption = agent.generate_caption(current_path)
        polished = professional_polish(final, last_suggestions)
        final_img = add_watermark(polished, caption)
        out_path = output_dir / f"{stem}_perfect.jpg"
        final_img.save(out_path, quality=95)
        logger.info(f"Saved: {out_path}")

def main():
    logger.info(f"Monitoring {input_dir}")
    while True:
        for p in input_dir.rglob("*.[jp][pn]g"):
            if not p.name.startswith(".") and not is_processed(p):
                process_photo(p)
        time.sleep(10)

if __name__ == "__main__":
    main()