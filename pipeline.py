"""
pipeline.py — Master Orchestrator
Runs the full document dewarping pipeline end-to-end.

Stages:
  1. Input validation
  2. Preprocessing (CLAHE, bilateral, Gaussian)
  3. Page detection (contour → quad)
  4. Perspective correction (homography)
  5. Document dewarping (coupled snakes)
  6. Image enhancement (sharpen, binarize)
  7. Quality analysis
  8. Export (image + PDF report)
"""
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def _safe_print(msg):
    try:
        print(msg)
    except Exception:
        try:
            print(str(msg).encode('ascii', errors='replace').decode('ascii'))
        except Exception:
            pass

from src.phase1_input       import load_image, validate_image
from src.phase2_preprocessing import preprocess
from src.phase3_page_detection import detect_page_quad, order_corners
from src.phase4_perspective  import correct_perspective
from src.phase5_dewarping    import dewarp
from src.phase6_enhancement  import enhance, remove_shadows, deskew_fine
from src.phase7_quality      import analyze_quality, QualityReport
from src.phase8_export       import (save_image, save_side_by_side,
                                      save_json_report, generate_pdf_report)


@dataclass
class PipelineResult:
    """Container for all pipeline outputs."""
    # Stage outputs
    original:          np.ndarray | None = None
    preprocessed:      dict              = field(default_factory=dict)
    quad:              np.ndarray | None = None
    quad_overlay:      np.ndarray | None = None
    perspective_corrected: np.ndarray | None = None
    dewarped:          np.ndarray | None = None
    enhanced:          dict              = field(default_factory=dict)
    dewarp_debug:      dict              = field(default_factory=dict)

    # Quality reports
    quality_before:    QualityReport | None = None
    quality_after:     QualityReport | None = None

    # Timing per stage (seconds)
    timings:           dict              = field(default_factory=dict)
    total_time:        float             = 0.0

    # Export paths
    saved_image_path:  Path | None      = None
    saved_pdf_path:    Path | None      = None
    saved_json_path:   Path | None      = None
    saved_comparison:  Path | None      = None

    # Status
    success:           bool             = True
    error_message:     str              = ""
    page_detected:     bool             = False


def run_pipeline(source,
                 output_dir: str | Path = "outputs",
                 save_outputs: bool = True,
                 run_dewarp: bool = True,
                 run_enhance: bool = True,
                 rotation: int = 0,
                 scan_mode: str = "clean_color",
                 trim_margin: float = 0.0,
                 ocr_lang: str = "eng",
                 verbose: bool = True) -> PipelineResult:
    """
    Execute the full document processing pipeline.

    Args:
        source:       Image path (str/Path), numpy array
        output_dir:   Root output directory
        save_outputs: Whether to save results to disk
        run_dewarp:   Run Phase 5 (dewarping). Set False for baseline mode.
        run_enhance:  Run Phase 6 (enhancement)
        rotation:     Manual rotation angle (0, 90, 180, 270 degrees CW)
        scan_mode:    "clean_color" | "clean_bw" | "dewarped_photo"
        trim_margin:  Fraction to trim from edges (e.g. 0.02 for hole punch trimming)
        ocr_lang:     Tesseract language code(s) e.g. "eng", "eng+hin"
        verbose:      Print progress to console

    Returns:
        PipelineResult with all stage outputs and metrics
    """
    result = PipelineResult()
    output_dir = Path(output_dir)
    t_total = time.perf_counter()

    def log(msg):
        if verbose:
            _safe_print(f"  {msg}")

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — Input
    # ══════════════════════════════════════════════════════════════════════════
    log("Phase 1: Loading image...")
    t0 = time.perf_counter()
    try:
        img = load_image(source, rotation=rotation)
        info = validate_image(img)
        if not info["valid"]:
            raise ValueError(f"Image too small: {info['width']}×{info['height']}")
        result.original = img.copy()
        log(f"         Loaded {info['width']}×{info['height']} px, {info['megapixels']} MP")
    except Exception as e:
        result.success = False
        result.error_message = f"Phase 1 failed: {e}"
        return result
    result.timings["input"] = time.perf_counter() - t0

    # ── Quality BEFORE ────────────────────────────────────────────────────────
    log("Phase 1: Analyzing pre-correction quality...")
    result.quality_before = analyze_quality(img)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2 — Preprocessing
    # ══════════════════════════════════════════════════════════════════════════
    log("Phase 2: Preprocessing (CLAHE, bilateral, Gaussian)...")
    t0 = time.perf_counter()
    result.preprocessed = preprocess(img)
    result.timings["preprocessing"] = time.perf_counter() - t0

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3 — Page Detection
    # ══════════════════════════════════════════════════════════════════════════
    log("Phase 3: Detecting document edges and quad...")
    t0 = time.perf_counter()
    quad, quad_overlay, paper_mask = detect_page_quad(img, draw=True)
    result.quad         = quad
    result.quad_overlay = quad_overlay
    result.page_detected = quad is not None
    result.timings["page_detection"] = time.perf_counter() - t0

    if not result.page_detected:
        log("         WARNING: No page quad found — using full image as input.")

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 4 — Perspective Correction (Zero Background Leakage)
    # ══════════════════════════════════════════════════════════════════════════
    log("Phase 4: Applying perspective correction (homography)...")
    t0 = time.perf_counter()
    if result.page_detected and quad is not None:
        try:
            perspective_img = correct_perspective(img, quad, paper_mask=paper_mask)
            # Sanity check: if corrected image is tiny, fall back
            if perspective_img is None or min(perspective_img.shape[:2]) < 50:
                log("         WARNING: Perspective result too small — using original image.")
                perspective_img = img.copy()
        except Exception as e:
            log(f"         WARNING: Perspective correction failed ({e}) — using original image.")
            perspective_img = img.copy()
    else:
        perspective_img = img.copy()
    result.perspective_corrected = perspective_img
    result.timings["perspective"] = time.perf_counter() - t0

    current = perspective_img

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 5 — Document Dewarping (Coupled Snakes)
    # ══════════════════════════════════════════════════════════════════════════
    if run_dewarp:
        log("Phase 5: Dewarping with coupled snakes...")
        t0 = time.perf_counter()
        dewarped, debug_info = dewarp(current, debug=True, auto_orient=False)
        result.dewarped     = dewarped
        result.dewarp_debug = debug_info
        current = dewarped
        n_lines = debug_info.get("n_lines", 0)
        log(f"         Detected {n_lines} text lines, displacement map applied.")
        result.timings["dewarping"] = time.perf_counter() - t0
    else:
        result.dewarped = current
        result.timings["dewarping"] = 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 6 — Image Enhancement (100% Scanning Effect)
    # ══════════════════════════════════════════════════════════════════════════
    if run_enhance:
        log("Phase 6: Enhancing image (surface flattening, sharpening, scan filter)...")
        t0 = time.perf_counter()

        result.enhanced = enhance(current, mode="full")
        result.timings["enhancement"] = time.perf_counter() - t0

    # ── Final output image selection ──────────────────────────────────────────
    if scan_mode == "clean_bw" and result.enhanced and "clean_bw" in result.enhanced:
        final_img = result.enhanced["clean_bw"]
    elif scan_mode == "dewarped_photo":
        final_img = current
    elif result.enhanced and "clean_color" in result.enhanced:
        final_img = result.enhanced["clean_color"]
    elif result.enhanced and "natural_scan" in result.enhanced:
        final_img = result.enhanced["natural_scan"]
    else:
        final_img = current

    if final_img.ndim == 2:
        final_img = cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)

    # Optional margin trimming (e.g. 2% to trim hole punches/binder rings)
    if trim_margin > 0.0:
        fh, fw = final_img.shape[:2]
        ty = int(fh * trim_margin)
        tx = int(fw * trim_margin)
        if fh - 2*ty > 50 and fw - 2*tx > 50:
            final_img = final_img[ty:fh-ty, tx:fw-tx]

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 7 — Quality Analysis (AFTER)
    # ══════════════════════════════════════════════════════════════════════════
    log("Phase 7: Analyzing post-correction quality...")
    t0 = time.perf_counter()
    result.quality_after = analyze_quality(final_img)
    result.timings["quality_analysis"] = time.perf_counter() - t0

    if verbose:
        print(result.quality_after)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 8 — Export
    # ══════════════════════════════════════════════════════════════════════════
    if save_outputs:
        log("Phase 8: Saving outputs...")
        t0 = time.perf_counter()

        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        corrected_dir = output_dir / "corrected"
        reports_dir   = output_dir / "reports"

        result.saved_image_path = save_image(
            final_img, corrected_dir, f"corrected_{ts}.png"
        )
        result.saved_comparison = save_side_by_side(
            img, final_img,
            output_dir / "comparisons" / f"comparison_{ts}.jpg"
        )
        result.saved_json_path = save_json_report(
            result.quality_after,
            reports_dir / f"report_{ts}.json"
        )
        result.saved_pdf_path = generate_pdf_report(
            result.quality_after,
            img,
            final_img,
            reports_dir / f"report_{ts}.pdf"
        )
        result.timings["export"] = time.perf_counter() - t0

    result.total_time = time.perf_counter() - t_total

    if verbose:
        _print_summary(result)

    return result


def _print_summary(result: PipelineResult):
    """Print a formatted pipeline summary."""
    print("\n" + "=" * 50)
    print("  PIPELINE COMPLETE")
    print("=" * 50)
    print(f"  Page detected  : {'OK' if result.page_detected else 'X'}")
    if result.quality_before and result.quality_after:
        delta = result.quality_after.overall_score - result.quality_before.overall_score
        print(f"  Quality Before : {result.quality_before.overall_score:.0f}/100")
        print(f"  Quality After  : {result.quality_after.overall_score:.0f}/100  "
              f"({'+' if delta >= 0 else '-'}{abs(delta):.1f})")
    print(f"  Total Time     : {result.total_time:.2f}s")
    print("  Timings:")
    for k, v in result.timings.items():
        print(f"    {k:<20}: {v*1000:.1f} ms")
    if result.saved_pdf_path:
        print(f"  Report saved   : {result.saved_pdf_path}")
    print("=" * 50)
