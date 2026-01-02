# sample.py (restored larger watermark)
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import cv2
import numpy as np
from pathlib import Path

INPUT_FILE = "sample.jpg"
OUTPUT_SUFFIX = "_static"

input_path = Path(INPUT_FILE)
stem = input_path.stem
suffix = input_path.suffix

img = Image.open(input_path)

# Enhancements
cropped = img.crop((0, 200, img.width, img.height - 100))
bright = ImageEnhance.Brightness(cropped).enhance(1.15)
contrasted = ImageEnhance.Contrast(bright).enhance(1.2)
img_np = np.array(contrasted)
gaussian = cv2.GaussianBlur(img_np, (9, 9), 10.0)
sharpened = cv2.addWeighted(img_np, 1.8, gaussian, -0.8, 0)
sharp_img = Image.fromarray(sharpened.clip(0, 255).astype(np.uint8))
lab = cv2.cvtColor(np.array(sharp_img), cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
l = clahe.apply(l)
enhanced_lab = cv2.merge([l, a, b])
final_np = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
final = Image.fromarray(final_np)

# Restored larger watermark
draw = ImageDraw.Draw(final)
font_size = int(final.height * 0.08)  # Larger size restored
try:
    font = ImageFont.truetype("arial.ttf", font_size)
except IOError:
    font = ImageFont.load_default()

text = "Perfect-Shot Static Sample"
bbox = draw.textbbox((0, 0), text, font=font)
w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
x, y = 30, final.height - h - 30

draw.text((x + 4, y + 4), text, fill="black", font=font)
draw.text((x, y), text, fill=(255, 255, 255, 220), font=font)

# Dynamic outputs
optimized = f"{stem}{OUTPUT_SUFFIX}{suffix}"
compare = f"{stem}{OUTPUT_SUFFIX}_compare{suffix}"

final.save(optimized)
combined = Image.new("RGB", (img.width + final.width, max(img.height, final.height)))
combined.paste(img, (0, 0))
combined.paste(final, (img.width, 0))
combined.save(compare)

print(f"Saved: {optimized} and {compare}")