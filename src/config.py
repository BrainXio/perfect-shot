# src/config.py
import os

VISION_MODEL = os.getenv("VISION_MODEL", "ministral-3:3b")
INPUT_DIR = os.getenv("INPUT_DIR", "/data/input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/data/output")
CACHE_DIR = os.getenv("CACHE_DIR", "/data/cache")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///mlruns")
AGENT_LANGUAGE = os.getenv("AGENT_LANGUAGE", "English")
LOGO_URL = os.getenv("LOGO_URL", "https://avatars.githubusercontent.com/u/164061086?v=4")
LOCAL_LOGO_PATH = os.getenv("LOCAL_LOGO_PATH", "/data/output/logo.png")