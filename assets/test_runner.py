"""
test_runner.py — Run page detection on complex_test.png
"""
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from src.phase3_page_detection import detect_page_quad

img = cv2.imread("assets/complex_test.png")
if img is None:
    print("Error: assets/complex_test.png not found")
    exit(1)

quad, annotated = detect_page_quad(img, draw=True)
print("--- DETECTED QUAD ---")
print(quad)

cv2.imwrite("assets/detection_output.png", annotated)
print("Saved output visualization to assets/detection_output.png")
