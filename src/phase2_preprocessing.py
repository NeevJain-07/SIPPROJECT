"""
Phase 2 — Preprocessing
Grayscale conversion, denoising, CLAHE, bilateral filtering.
"""
import cv2
import numpy as np


def preprocess(img: np.ndarray, debug: bool = False) -> dict:
    """
    Full preprocessing pipeline.
    
    Returns:
        dict with keys: gray, denoised, clahe, edges_ready
    """
    steps = {}
    
    # ── Step 1: Grayscale ─────────────────────────────────────────────────────
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    steps["gray"] = gray
    
    # ── Step 2: Bilateral filter / Edge-preserving smoothing ─────────────
    if max(gray.shape) > 2000:
        bilateral = cv2.GaussianBlur(gray, (5, 5), 1.2)
    else:
        bilateral = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    steps["bilateral"] = bilateral
    
    # ── Step 3: Gaussian blur for edge detection prep ─────────────────────────
    blurred = cv2.GaussianBlur(bilateral, (5, 5), 0)
    steps["blurred"] = blurred
    
    # ── Step 4: CLAHE — Contrast Limited Adaptive Histogram Equalization ──────
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe = clahe_obj.apply(bilateral)
    steps["clahe"] = clahe
    
    # ── Step 5: Fast edge-ready image ─────────────────────────────────────────
    steps["denoised"] = bilateral
    steps["edges_ready"] = blurred
    
    if debug:
        _show_preprocessing_steps(steps)
    
    return steps


def auto_threshold_canny(img_gray: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Compute Canny thresholds automatically using Otsu's method.
    
    Returns:
        (edges, low_threshold, high_threshold)
    """
    # Otsu threshold on the blurred image
    _, otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Canny thresholds derived from Otsu
    high = float(otsu.mean()) if otsu.mean() > 0 else 128.0
    low = high * 0.33
    
    edges = cv2.Canny(img_gray, low, high)
    return edges, low, high


def _show_preprocessing_steps(steps: dict):
    """Debug visualization of preprocessing stages."""
    import matplotlib.pyplot as plt
    
    names = ["gray", "bilateral", "clahe", "denoised"]
    fig, axes = plt.subplots(1, len(names), figsize=(20, 5))
    fig.suptitle("Preprocessing Pipeline", fontsize=14)
    
    for ax, name in zip(axes, names):
        if name in steps:
            ax.imshow(steps[name], cmap="gray")
            ax.set_title(name.capitalize())
            ax.axis("off")
    
    plt.tight_layout()
    plt.show()
