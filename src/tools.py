# src/tools.py
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import cv2
import numpy as np
from pathlib import Path

def crop(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    return image.crop(bbox)

def adjust_exposure(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)

def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(image).enhance(factor)

def sharpen(image: Image.Image, factor: float) -> Image.Image:
    img_np = np.array(image)
    gaussian = cv2.GaussianBlur(img_np, (9, 9), 10.0)
    sharpened = cv2.addWeighted(img_np, 1 + factor, gaussian, -factor, 0)
    return Image.fromarray(sharpened.clip(0, 255).astype(np.uint8))

def auto_enhance(image: Image.Image) -> Image.Image:
    img_np = np.array(image.convert("LAB"))
    l, a, b = cv2.split(img_np)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB))

def compare(before: Image.Image, after: Image.Image) -> Image.Image:
    width = before.width + after.width
    height = max(before.height, after.height)
    combined = Image.new("RGB", (width, height))
    combined.paste(before, (0, 0))
    combined.paste(after, (before.width, 0))
    return combined

def add_watermark(image: Image.Image, text: str = "Perfect-Shot") -> Image.Image:
    draw = ImageDraw.Draw(image)
    font_size = int(image.height * 0.08)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x, y = 30, image.height - h - 30

    draw.text((x + 4, y + 4), text, fill="black", font=font)
    draw.text((x, y), text, fill=(255, 255, 255, 220), font=font)
    return image

def get_opencv_metrics(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        return {}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    noise = float(np.std(gray))
    contrast = float(np.sqrt(np.mean((gray.astype(float) - np.mean(gray))**2)))
    exposure = float(np.mean(gray))
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = float(np.mean(hsv[:,:,1]))
    
    hist = cv2.calcHist([gray], [0], None, [256], [0,256]).flatten()
    cumsum = np.cumsum(hist)
    cumsum /= cumsum[-1] + 1e-8
    low = int(np.argmax(cumsum > 0.05))
    high = int(np.argmax(cumsum > 0.95))
    dynamic_range = high - low
    
    return {
        "sharpness": round(sharpness, 2),
        "noise": round(noise, 2),
        "contrast": round(contrast, 2),
        "exposure": round(exposure, 2),
        "saturation": round(saturation, 2),
        "dynamic_range": dynamic_range
    }

def save_tweaked(original_path: str, tweaked_image: Image.Image) -> str:
    output_path = Path(original_path).with_name(f"{Path(original_path).stem}_tweaked.jpg")
    tweaked_image.save(output_path)
    return str(output_path)