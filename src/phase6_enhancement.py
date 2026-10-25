"""
Phase 6 — Image Enhancement
Sharpening, adaptive binarization, Lanczos interpolation, background removal.
"""
import cv2
import numpy as np


def enhance(img: np.ndarray, mode: str = "full") -> dict:
    """
    Apply post-dewarping image enhancements.

    Args:
        img:  BGR document image (already dewarped)
        mode: "full" | "sharpen_only" | "binary_only"

    Returns:
        dict of enhanced images keyed by method name
    """
    results = {}

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # ── 1. Unsharp Masking (Sharpening) ──────────────────────────────────────
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    results["sharpened"] = sharpened

    if mode == "sharpen_only":
        return results

    # ── 2. Adaptive Threshold Binarization ───────────────────────────────────
    binary = cv2.adaptiveThreshold(
        sharpened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    results["binary"] = binary

    # ── 3. Morphological Cleanup (remove small noise blobs) ──────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    results["cleaned_binary"] = cleaned

    if mode == "binary_only":
        return results

    # ── 4. CLAHE on grayscale (for grayscale output) ──────────────────────────
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    results["clahe_gray"] = enhanced_gray

    # ── 5. Full enhanced color image ──────────────────────────────────────────
    if len(img.shape) == 3:
        # Fast uint8 unsharp masking
        blur_color = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
        sharp_color = cv2.addWeighted(img, 1.35, blur_color, -0.35, 0)
        results["sharpened_color"] = sharp_color
        results["background_normalized"] = enhanced_gray

        # ── 6. "Clean Color Scan" (CamScanner-Style Flatbed Color Scan) ───────
        lab = cv2.cvtColor(sharp_color, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lh, lw = l.shape[:2]

        # Multi-scale fast background estimation
        scale = 1000.0 / max(lh, lw) if max(lh, lw) > 1000 else 1.0
        small_l = cv2.resize(l, (int(lw * scale), int(lh * scale))) if scale != 1.0 else l.copy()
        k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        bg_small = cv2.morphologyEx(small_l, cv2.MORPH_CLOSE, k_small)
        bg_small = cv2.GaussianBlur(bg_small, (21, 21), 0)
        bg_l = cv2.resize(bg_small, (lw, lh), interpolation=cv2.INTER_LINEAR) if scale != 1.0 else bg_small

        norm_l = cv2.divide(l.astype(np.float32), bg_l.astype(np.float32) + 1e-5, scale=255.0)
        norm_l = np.clip(norm_l, 0, 255)
        p1, p95 = np.percentile(norm_l, (1, 95))
        norm_l = np.clip((norm_l - p1) * (255.0 / max(p95 - p1, 1e-5)), 0, 255).astype(np.uint8)

        # Unsharp mask for text definition
        blur_l = cv2.GaussianBlur(norm_l, (0, 0), sigmaX=2.0)
        sharp_l = cv2.addWeighted(norm_l, 1.4, blur_l, -0.4, 0)

        # Subtle chroma pop for colored ink
        a_boost = cv2.addWeighted(a, 1.15, np.full_like(a, 128), -0.15, 0)
        b_boost = cv2.addWeighted(b, 1.15, np.full_like(b, 128), -0.15, 0)

        clean_color = cv2.cvtColor(cv2.merge([sharp_l, a_boost, b_boost]), cv2.COLOR_LAB2BGR)
        results["clean_color"] = clean_color
        results["magic_color"] = clean_color
        results["natural_scan"] = clean_color

        # ── 8. "Clean B&W Scan" (Pristine High-Contrast Document Scan) ────────
        _, bw = cv2.threshold(sharp_l, 205, 255, cv2.THRESH_BINARY)
        # Fast morphological speckle noise cleanup
        clean_bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        clean_bw_bgr = cv2.cvtColor(clean_bw, cv2.COLOR_GRAY2BGR)
        results["clean_bw"] = clean_bw_bgr
    else:
        sharp_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        bin_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        results["sharpened_color"] = sharp_bgr
        results["clean_color"] = sharp_bgr
        results["clean_bw"] = bin_bgr
        results["natural_scan"] = sharp_bgr
        results["magic_color"] = sharp_bgr

    return results


def remove_shadows(img: np.ndarray) -> np.ndarray:
    """
    Remove uneven lighting / shadows from scanned document.
    Uses morphological background estimation.
    """
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Estimate background illumination
        dilated = cv2.dilate(l, np.ones((21, 21), np.uint8), iterations=2)
        bg = cv2.GaussianBlur(dilated, (91, 91), 0)

        # Normalize: remove low-freq illumination component
        normalized = cv2.divide(l.astype(np.float32),
                                 bg.astype(np.float32),
                                 scale=255).astype(np.uint8)

        lab[:, :, 0] = normalized
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        l = img.copy()

        # Estimate background illumination
        dilated = cv2.dilate(l, np.ones((21, 21), np.uint8), iterations=2)
        bg = cv2.GaussianBlur(dilated, (91, 91), 0)

        # Normalize: remove low-freq illumination component
        result = cv2.divide(l.astype(np.float32),
                             bg.astype(np.float32),
                             scale=255).astype(np.uint8)

    return result


def super_resolve(img: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """
    Simple bicubic upscaling as a lightweight "super resolution" step.
    Improves readability of low-resolution scans.
    """
    h, w = img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    upscaled = cv2.resize(img, (new_w, new_h),
                           interpolation=cv2.INTER_LANCZOS4)
    return upscaled


def deskew_fine(img_gray: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Fine-tune skew correction using Hough transform on text lines.
    Rotates the image by the estimated skew angle.

    Returns:
        (deskewed_image, skew_angle_degrees)
    """
    edges = cv2.Canny(img_gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=100,
                             minLineLength=img_gray.shape[1] // 4,
                             maxLineGap=30)

    if lines is None:
        return img_gray, 0.0

    angles = []
    for line in lines:
        # OpenCV 5 returns (N,4); OpenCV 4 returns (N,1,4) — flatten handles both
        x1, y1, x2, y2 = line.flatten()[:4]
        if x2 != x1:
            angle = np.degrees(np.arctan2(int(y2) - int(y1), int(x2) - int(x1)))
            # Only consider near-horizontal lines (text lines)
            if abs(angle) < 15:
                angles.append(angle)

    if not angles:
        return img_gray, 0.0

    median_angle = float(np.median(angles))

    h, w = img_gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    deskewed = cv2.warpAffine(img_gray, M, (w, h),
                               flags=cv2.INTER_LANCZOS4,
                               borderMode=cv2.BORDER_REPLICATE)

    return deskewed, median_angle
