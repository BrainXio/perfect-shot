# src/logger.py
import logging
import sys

logger = logging.getLogger("perfect-shot")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
))
logger.addHandler(handler)