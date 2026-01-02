# src/tools.py
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import cv2
import numpy as np

def crop(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    return image.crop(bbox)

def adjust_exposure(image: Image.Image, factor: float) -> Image.Image:
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

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

def add_watermark(image: Image.Image, text: str = "Perfect-Shot Static Sample") -> Image.Image:
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