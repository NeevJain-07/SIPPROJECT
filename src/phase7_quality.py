"""
Phase 7 — Document Quality Analyzer
Evaluates scanned document quality across multiple metrics and generates
a structured quality report with an overall score and star rating.

Metrics:
  - Sharpness       (Variance of Laplacian)
  - Blur Level      (Low / Medium / High)
  - Brightness      (Too Dark / Good / Too Bright)
  - Contrast        (RMS contrast)
  - Perspective     (Edge parallelism check)
  - Noise Level     (High-frequency residual estimation)
  - OCR Readiness   (Composite prediction)
  - Overall Score   (/100) + Star Rating
"""
import cv2
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Literal


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QualityReport:
    # Raw metric values
    sharpness_score: float = 0.0        # 0–100
    blur_variance:   float = 0.0        # raw Laplacian variance
    brightness_mean: float = 0.0        # 0–255
    contrast_rms:    float = 0.0        # 0–127
    noise_estimate:  float = 0.0        # 0–100 (lower = less noise)

    # Classification labels
    blur_level:    Literal["Low", "Medium", "High"] = "Medium"
    brightness_ok: Literal["Too Dark", "Good", "Too Bright"] = "Good"
    contrast_ok:   Literal["Low", "Good", "Excellent"] = "Good"
    noise_level:   Literal["Low", "Medium", "High"] = "Medium"
    perspective_ok: bool = True

    # Composite scores
    ocr_readiness:  float = 0.0         # 0–100
    overall_score:  float = 0.0         # 0–100
    star_rating:    int   = 3           # 1–5

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        stars = "★" * self.star_rating + "☆" * (5 - self.star_rating)
        lines = [
            "╔══════════════════════════════════════════╗",
            "║        DOCUMENT QUALITY REPORT           ║",
            "╚══════════════════════════════════════════╝",
            f"  Sharpness      : {self.sharpness_score:.1f}%",
            f"  Blur           : {self.blur_level}",
            f"  Brightness     : {self.brightness_ok}",
            f"  Contrast       : {self.contrast_ok}",
            f"  Perspective    : {'Correct' if self.perspective_ok else 'Distorted'}",
            f"  Noise Level    : {self.noise_level}",
            f"  OCR Readiness  : {self.ocr_readiness:.1f}%",
            "  ──────────────────────────────────────────",
            f"  Overall Score  : {self.overall_score:.0f} / 100",
            f"  Rating         : {stars}",
        ]
        if self.recommendations:
            lines.append("  ──────────────────────────────────────────")
            lines.append("  Recommendations:")
            for r in self.recommendations:
                lines.append(f"    • {r}")
        lines.append("  ══════════════════════════════════════════")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Individual Metric Functions
# ══════════════════════════════════════════════════════════════════════════════

def measure_sharpness(gray: np.ndarray) -> tuple[float, float]:
    """
    Sharpness via Variance of Laplacian.
    High variance = sharp; Low variance = blurry.

    Returns:
        (score_0_100, raw_variance)
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(lap.var())

    # Empirical calibration:
    # variance < 50   → very blurry
    # variance > 2000 → very sharp
    score = np.clip(variance / 2000.0 * 100, 0, 100)
    return float(score), variance


def classify_blur(variance: float) -> Literal["Low", "Medium", "High"]:
    """Classify blur level from Laplacian variance."""
    if variance > 500:
        return "Low"
    elif variance > 100:
        return "Medium"
    else:
        return "High"


def measure_brightness(gray: np.ndarray) -> tuple[float, Literal["Too Dark", "Good", "Too Bright"]]:
    """
    Measure mean brightness and classify.

    Returns:
        (mean_brightness, label)
    """
    mean = float(gray.mean())
    if mean < 60:
        label: Literal["Too Dark", "Good", "Too Bright"] = "Too Dark"
    elif mean > 200:
        label = "Too Bright"
    else:
        label = "Good"
    return mean, label


def measure_contrast(gray: np.ndarray) -> tuple[float, Literal["Low", "Good", "Excellent"]]:
    """
    RMS contrast = std(pixel_intensities).

    Returns:
        (rms_contrast, label)
    """
    rms = float(gray.std())
    if rms < 30:
        label: Literal["Low", "Good", "Excellent"] = "Low"
    elif rms > 60:
        label = "Excellent"
    else:
        label = "Good"
    return rms, label


def estimate_noise(gray: np.ndarray) -> tuple[float, Literal["Low", "Medium", "High"]]:
    """
    Estimate noise level using high-frequency residual.
    noise_img = |image - gaussian_smoothed|
    noise_estimate = mean of noise_img

    Returns:
        (noise_score_0_100, label)
    """
    smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = cv2.absdiff(gray, smoothed).astype(np.float32)
    noise_mean = float(residual.mean())

    # Normalize: 0 = no noise, 100 = very noisy
    score = np.clip(noise_mean / 30.0 * 100, 0, 100)

    if noise_mean < 5:
        label: Literal["Low", "Medium", "High"] = "Low"
    elif noise_mean < 12:
        label = "Medium"
    else:
        label = "High"

    return float(score), label


def check_perspective(gray: np.ndarray) -> bool:
    """
    Check edge parallelism after perspective correction.
    Uses Hough lines to detect the dominant near-vertical and near-horizontal
    lines and checks their variance in angle.

    Returns True if perspective appears correct (low angle variance).
    """
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80,
        minLineLength=gray.shape[1] // 5, maxLineGap=20
    )
    if lines is None or len(lines) < 4:
        return True  # Not enough lines to evaluate

    angles = []
    for line in lines:
        # OpenCV 5 returns (N,4); OpenCV 4 returns (N,1,4) — flatten handles both
        x1, y1, x2, y2 = line.flatten()[:4]
        if x2 != x1:
            angles.append(np.degrees(np.arctan2(int(y2) - int(y1), int(x2) - int(x1))))

    if not angles:
        return True

    # Separate near-horizontal lines
    h_angles = [a for a in angles if abs(a) < 20]
    if len(h_angles) < 2:
        return True

    # Low std of horizontal angles → parallel lines → good perspective
    std = float(np.std(h_angles))
    return std < 5.0


def compute_ocr_readiness(sharpness: float, blur: str,
                            brightness: str, contrast: str,
                            noise: str) -> float:
    """
    Composite OCR readiness prediction (0–100).
    Weighted combination of quality metrics.
    """
    score = sharpness * 0.35  # Sharpness matters most for OCR

    if blur == "Low":
        score += 25.0
    elif blur == "Medium":
        score += 12.0
    # High blur: +0

    if brightness == "Good":
        score += 20.0
    elif brightness in ("Too Dark", "Too Bright"):
        score += 5.0

    if contrast == "Excellent":
        score += 15.0
    elif contrast == "Good":
        score += 8.0

    if noise == "Low":
        score += 5.0
    elif noise == "Medium":
        score += 2.0

    return float(np.clip(score, 0, 100))


def compute_overall_score(sharpness: float, blur: str,
                           brightness: str, contrast: str,
                           noise: str, perspective_ok: bool) -> float:
    """
    Overall document quality score (0–100).
    """
    score = 0.0

    # Sharpness: 30 points
    score += sharpness * 0.30

    # Blur: 20 points
    blur_pts = {"Low": 20, "Medium": 10, "High": 0}
    score += blur_pts.get(blur, 0)

    # Brightness: 15 points
    score += 15 if brightness == "Good" else 4

    # Contrast: 15 points
    contrast_pts = {"Excellent": 15, "Good": 10, "Low": 2}
    score += contrast_pts.get(contrast, 0)

    # Noise: 10 points
    noise_pts = {"Low": 10, "Medium": 5, "High": 0}
    score += noise_pts.get(noise, 0)

    # Perspective: 10 points
    score += 10 if perspective_ok else 0

    return float(np.clip(score, 0, 100))


def score_to_stars(score: float) -> int:
    """Convert 0–100 score to 1–5 star rating."""
    if score >= 90:
        return 5
    elif score >= 75:
        return 4
    elif score >= 55:
        return 3
    elif score >= 35:
        return 2
    else:
        return 1


def generate_recommendations(blur: str, brightness: str,
                               contrast: str, noise: str,
                               perspective_ok: bool) -> list[str]:
    """Generate actionable auto-fix recommendations."""
    recs = []
    if blur in ("Medium", "High"):
        recs.append("Apply sharpening filter to improve text clarity.")
    if brightness == "Too Dark":
        recs.append("Increase brightness — image is underexposed.")
    if brightness == "Too Bright":
        recs.append("Reduce brightness / exposure — image is overexposed.")
    if contrast == "Low":
        recs.append("Enhance contrast using CLAHE or histogram equalization.")
    if noise in ("Medium", "High"):
        recs.append("Apply denoising (bilateral filter or NLM) to reduce noise.")
    if not perspective_ok:
        recs.append("Recapture with the document flat on a surface.")
    if not recs:
        recs.append("Document quality is excellent — no corrections needed.")
    return recs


# ══════════════════════════════════════════════════════════════════════════════
# Master Quality Analyzer
# ══════════════════════════════════════════════════════════════════════════════

def analyze_quality(img: np.ndarray) -> QualityReport:
    """
    Run the full quality analysis pipeline on a document image.

    Args:
        img: BGR or grayscale document image

    Returns:
        QualityReport dataclass with all metrics
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # ── Collect metrics ───────────────────────────────────────────────────────
    sharpness_score, blur_var   = measure_sharpness(gray)
    blur_level                   = classify_blur(blur_var)
    brightness_mean, brightness  = measure_brightness(gray)
    contrast_rms, contrast_label = measure_contrast(gray)
    noise_score, noise_label     = estimate_noise(gray)
    perspective_ok               = check_perspective(gray)

    # ── Compute composites ────────────────────────────────────────────────────
    ocr_readiness = compute_ocr_readiness(
        sharpness_score, blur_level, brightness, contrast_label, noise_label
    )
    overall_score = compute_overall_score(
        sharpness_score, blur_level, brightness, contrast_label,
        noise_label, perspective_ok
    )
    star_rating = score_to_stars(overall_score)
    recs = generate_recommendations(
        blur_level, brightness, contrast_label, noise_label, perspective_ok
    )

    report = QualityReport(
        sharpness_score  = round(sharpness_score, 2),
        blur_variance    = round(blur_var, 2),
        brightness_mean  = round(brightness_mean, 2),
        contrast_rms     = round(contrast_rms, 2),
        noise_estimate   = round(noise_score, 2),
        blur_level       = blur_level,
        brightness_ok    = brightness,
        contrast_ok      = contrast_label,
        noise_level      = noise_label,
        perspective_ok   = perspective_ok,
        ocr_readiness    = round(ocr_readiness, 2),
        overall_score    = round(overall_score, 2),
        star_rating      = star_rating,
        recommendations  = recs,
    )

    return report
