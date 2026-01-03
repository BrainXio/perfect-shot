# src/tools.py (fixed)
import requests
from io import BytesIO
import cv2, numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from pathlib import Path
from logger import logger
from config import LOGO_URL
from models import EnhancementSuggestions

def crop(img: Image.Image, bbox): 
    return img.crop(bbox)

def adjust_exposure(img: Image.Image, f: float): 
    return ImageEnhance.Brightness(img).enhance(f)

def adjust_contrast(img: Image.Image, f: float): 
    return ImageEnhance.Contrast(img).enhance(f)

def sharpen(img: Image.Image, f: float):
    a = np.array(img)
    blur = cv2.GaussianBlur(a, (9,9), 10)
    return Image.fromarray(cv2.addWeighted(a, 1+f, blur, -f, 0).clip(0,255).astype(np.uint8))

def auto_enhance(img: Image.Image):
    lab = np.array(img.convert("LAB"))
    l = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8)).apply(lab[:,:,0])
    lab[:,:,0] = l
    return Image.fromarray(cv2.cvtColor(cv2.merge(lab[:,:,::-1]), cv2.COLOR_LAB2RGB))

def compare(before: Image.Image, after: Image.Image):
    w = before.width + after.width
    h = max(before.height, after.height)
    comb = Image.new("RGB", (w, h))
    comb.paste(before, (0,0))
    comb.paste(after, (before.width,0))
    return comb

def add_vignette(img: Image.Image, strength: float = 0.3):
    w, h = img.size
    x = np.linspace(-1,1,w)
    y = np.linspace(-1,1,h)
    X,Y = np.meshgrid(x,y)
    radius = np.sqrt(X**2 + Y**2)
    mask = 1 - np.clip((radius-0.8)/(1.4-0.8),0,1)**2
    mask = 1 - strength * (1-mask)
    mask_img = Image.fromarray((mask*255).astype(np.uint8)).resize(img.size)
    dark = Image.new("RGB", img.size)
    return Image.blend(img, Image.composite(dark, img, mask_img), strength)

def professional_polish(img: Image.Image, sug: EnhancementSuggestions):
    if sug.color_boost: 
        img = ImageEnhance.Color(img).enhance(sug.color_boost)
    if sug.vignette_strength: 
        img = add_vignette(img, sug.vignette_strength)
    if not sug.sharpen_factor or sug.sharpen_factor < 0.3: 
        img = sharpen(img, 0.3)
    return img

def add_watermark(img: Image.Image, caption: str = ""):
    try:
        logo = Image.open(BytesIO(requests.get(LOGO_URL).content)).convert("RGBA")
        data = np.array(logo)
        data[(data[:,:,1]>150)&(data[:,:,0]<120)&(data[:,:,2]<120),3] = 0
        logo = Image.fromarray(data)
    except Exception as e:
        logger.warning(f"Logo fetch failed: {e}")
        return img

    logo.thumbnail((min(img.size)//10,))
    pos = (20, img.height - logo.height - 20)

    if caption:
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        bbox = draw.textbbox((0,0), caption, font=font)
        text_pos = (pos[0] + logo.width + 15, pos[1] + (logo.height - (bbox[3]-bbox[1]))//2)
        for dx,dy in [(x,y) for x in (-2,0,2) for y in (-2,0,2) if (x,y)!=(0,0)]:
            draw.text((text_pos[0]+dx, text_pos[1]+dy), caption, font=font, fill="black")
        draw.text(text_pos, caption, font=font, fill="white")

    img.convert("RGBA").paste(logo, pos, logo)
    return img.convert("RGB")

def get_opencv_metrics(path: str) -> dict:
    img = cv2.imread(path)
    if img is None: 
        return {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).flatten()
    cum = np.cumsum(hist)
    cum /= cum[-1] + 1e-8
    low = int(np.argmax(cum > 0.05))
    high = int(np.argmax(cum > 0.95))
    return {
        "sharpness": round(cv2.Laplacian(gray, cv2.CV_64F).var(), 2),
        "noise": round(np.std(gray), 2),
        "contrast": round(np.sqrt(np.mean((gray.astype(float)-gray.mean())**2)), 2),
        "exposure": round(gray.mean(), 2),
        "saturation": round(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,1].mean(), 2),
        "dynamic_range": high - low
    }