# src/tools.py
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import cv2
import numpy as np
from pathlib import Path
from logger import logger
from config import LOCAL_LOGO_PATH
import requests
from io import BytesIO
from config import LOGO_URL

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

def add_vignette(image: Image.Image, strength: float = 0.4) -> Image.Image:
    """Subtle dark vignette on corners/edges"""
    width, height = image.size
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    radius = np.sqrt(X**2 + Y**2)
    mask = 1 - np.clip((radius - 0.8) / (1.4 - 0.8), 0, 1)**2
    mask = 1 - strength * (1 - mask)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.resize(image.size, Image.BICUBIC)
    dark = Image.new("RGB", image.size, (0, 0, 0))
    vignette = Image.composite(dark, image, mask_img)
    return Image.blend(image, vignette, strength)

def professional_polish(image: Image.Image) -> Image.Image:
    """Final professional polish: subtle vignette, color enhancement, gentle sharpen"""
    img = image.copy()
    
    # Subtle color boost
    img = ImageEnhance.Color(img).enhance(1.08)
    
    # Gentle contrast
    img = ImageEnhance.Contrast(img).enhance(1.05)
    
    # Subtle vignette
    img = add_vignette(img, strength=0.35)
    
    # Very light final sharpen
    img = sharpen(img, factor=0.3)
    
    return img

# src/tools.py (updated add_watermark only)

def add_watermark(image: Image.Image) -> Image.Image:
    try:
        response = requests.get(LOGO_URL, timeout=10)
        response.raise_for_status()
        logo = Image.open(BytesIO(response.content)).convert("RGBA")
    except Exception as e:
        logger.warning(f"Failed to fetch logo: {e}")
        return image

    data = np.array(logo)
    # Make bright green (high G, low R/B) transparent
    mask = (data[:,:,1] > 150) & (data[:,:,0] < 120) & (data[:,:,2] < 120)
    data[mask, 3] = 0  # Set alpha to 0
    logo = Image.fromarray(data)

    max_size = min(image.width, image.height) // 10
    logo.thumbnail((max_size, max_size), Image.LANCZOS)

    # Full opacity for remaining parts
    if logo.mode == "RGBA":
        alpha = logo.split()[3]
        alpha = ImageEnhance.Brightness(alpha).enhance(1.0)
        logo.putalpha(alpha)

    # Bottom-left corner
    margin = 20
    position = (margin, image.height - logo.height - margin)

    rgba_image = image.convert("RGBA")
    rgba_image.paste(logo, position, logo)
    return rgba_image.convert("RGB")

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