# src/process.py
from agent import PhotoAgent
from logger import logger
from config import INPUT_DIR, OUTPUT_DIR, CACHE_DIR, MLFLOW_TRACKING_URI
from pathlib import Path
from PIL import Image
import time
import json
import mlflow
from tools import compare, add_watermark, professional_polish

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

agent = PhotoAgent()

input_dir = Path(INPUT_DIR)
output_dir = Path(OUTPUT_DIR)
cache_dir = Path(CACHE_DIR)
output_dir.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)

def is_processed(input_path: Path) -> bool:
    return (output_dir / f"{input_path.stem}_perfect.jpg").exists()

def process_photo(input_path: Path):
    logger.info(f"PROCESSING: {input_path.name}")
    original_path = str(input_path)
    current_path = original_path
    stem = input_path.stem
    iteration = 0
    max_iterations = 3

    with mlflow.start_run(run_name=f"{stem}_{time.strftime('%Y%m%d-%H%M%S')}"):
        mlflow.log_param("input_file", str(input_path))
        mlflow.log_param("vision_model", "ministral-3:3b")

        while True:
            analysis = agent.analyze_image(current_path)
            json_path = cache_dir / f"{stem}_analysis_v{iteration}.json"
            with open(json_path, "w") as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"SAVED TO CACHE: {json_path.name} (grade {analysis['grade']}/10)")

            mlflow.log_metric("grade", analysis["grade"], step=iteration)
            mlflow.log_artifact(str(json_path))

            grade = analysis["grade"]
            if grade >= 9.0 or iteration >= max_iterations:
                break

            suggestions = agent.get_enhancement_suggestions(analysis, current_path)
            has_suggestions = (
                suggestions.crop_bbox is not None or
                suggestions.auto_enhance is True or
                (suggestions.exposure_factor is not None and suggestions.exposure_factor != 1.0) or
                (suggestions.contrast_factor is not None and suggestions.contrast_factor != 1.0) or
                (suggestions.sharpen_factor is not None and suggestions.sharpen_factor > 0)
            )
            if not has_suggestions:
                logger.info("No meaningful suggestions")
                break

            enhanced_img = agent.apply_enhancements(current_path, suggestions)
            iteration += 1
            enhanced_path = cache_dir / f"{stem}_enhanced_v{iteration}.jpg"
            enhanced_img.save(enhanced_path, quality=95)
            logger.info(f"SAVED TO CACHE: {enhanced_path.name}")
            mlflow.log_artifact(str(enhanced_path))

            before_img = Image.open(current_path)
            comp_img = compare(before_img, enhanced_img)
            comp_path = cache_dir / f"{stem}_compare_v{iteration}.jpg"
            comp_img.save(comp_path, quality=95)
            logger.info(f"SAVED TO CACHE: {comp_path.name}")
            mlflow.log_artifact(str(comp_path))

            current_path = str(enhanced_path)

        # Final professional polish + watermark
        final_img = Image.open(current_path)
        polished_img = professional_polish(final_img)
        polished_img = add_watermark(polished_img)
        perfect_path = output_dir / f"{stem}_perfect.jpg"
        polished_img.save(perfect_path, quality=95)
        logger.info(f"SAVED FINAL TO OUTPUT: {perfect_path.name}")
        mlflow.log_artifact(str(perfect_path))

        if iteration > 0:
            orig_img = Image.open(original_path)
            final_comp = compare(orig_img, polished_img)
            final_comp_path = output_dir / f"{stem}_compare_final.jpg"
            final_comp.save(final_comp_path, quality=95)
            logger.info(f"SAVED FINAL COMPARE TO OUTPUT: {final_comp_path.name}")
            mlflow.log_artifact(str(final_comp_path))

        # Copy last analysis to output
        last_analysis = cache_dir / f"{stem}_analysis_v{iteration}.json"
        if last_analysis.exists():
            final_analysis_path = output_dir / f"{stem}_analysis.json"
            final_analysis_path.write_bytes(last_analysis.read_bytes())
            mlflow.log_artifact(str(final_analysis_path))

def main():
    logger.info(f"Monitoring {input_dir}")
    while True:
        found = False
        for img_path in input_dir.rglob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"} and not img_path.name.startswith(".") and not is_processed(img_path):
                found = True
                process_photo(img_path)
        if not found:
            logger.info("No unprocessed images")
        time.sleep(10)

if __name__ == "__main__":
    main()