# src/tools.py
from PIL import Image, ImageEnhance, ImageOps
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
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)

def auto_enhance(image: Image.Image) -> Image.Image:
    return ImageOps.autocontrast(image)

def compare(before: Image.Image, after: Image.Image) -> Image.Image:
    width = before.width + after.width
    height = max(before.height, after.height)
    combined = Image.new("RGB", (width, height))
    combined.paste(before, (0, 0))
    combined.paste(after, (before.width, 0))
    return combined