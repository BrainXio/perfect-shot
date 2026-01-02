# src/config.py
import os

VISION_MODEL = os.getenv("VISION_MODEL", "ministral-3:3b")
INPUT_DIR = os.getenv("INPUT_DIR", "/data/input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/data/output")
AGENT_LANGUAGE = os.getenv("AGENT_LANGUAGE", "English")
