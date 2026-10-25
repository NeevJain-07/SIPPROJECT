"""
Phase 5 — Document Dewarping
Implements coupled-snake based dewarping inspired by:
  Bukhari, Shafait, Breuel (2009) — "Dewarping of Document Images using Coupled-Snakes"

Pipeline:
  1. Binarize document (CLAHE + Otsu + Adaptive)
  2. Detect text lines via horizontal projection profile
  3. Fit coupled snakes (upper + lower contour) to each text line
  4. Estimate page curvature model from snake midlines
  5. Build displacement map (bivariate spline / griddata fallback)
  6. Remap image using cv2.remap()
"""
import cv2
import numpy as np
from scipy.interpolate import RectBivariateSpline, griddata
from scipy.ndimage import map_coordinates


# ══════════════════════════════════════════════════════════════════════════════
# 1. Binarization
# ══════════════════════════════════════════════════════════════════════════════

def binarize(img: np.ndarray) -> np.ndarray:
    """
    Convert to binary document image.
    Uses CLAHE normalisation + Otsu + adaptive thresholding combined.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Normalize contrast first — helps with uneven lighting / shadows
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Otsu global binarization
    _, otsu = cv2.threshold(gray, 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Adaptive threshold for local brightness variation
    adaptive = cv2.adaptiveThreshold(gray, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 25, 8)

    # Combine: pixel is foreground if EITHER method says so
    binary = cv2.bitwise_or(otsu, adaptive)

    # Morphological cleanup: remove tiny isolated specks
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    return binary


# ══════════════════════════════════════════════════════════════════════════════
# 2. Text-Line Detection via Morphological Smearing & Connected Components
# ══════════════════════════════════════════════════════════════════════════════

def detect_text_lines(binary_or_gray: np.ndarray,
                      min_line_height: int = 5,
                      min_gap: int = 4) -> list[tuple[int, int]]:
    """
    Detect text lines using horizontal morphological smearing and connected components.
    Overcomes limitations of global projection profiles on noisy/lined pages.

    Args:
        binary_or_gray:  Binary image or Grayscale image
        min_line_height: Minimum pixel rows for a valid text line
        min_gap:         Minimum gap rows between lines

    Returns:
        List of (y_start, y_end) tuples for each detected text line
    """
    if len(binary_or_gray.shape) == 3:
        gray = cv2.cvtColor(binary_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = binary_or_gray

    h, w = gray.shape[:2]

    # Inverted binary image: ink/lines = 255, paper = 0
    if gray.dtype == np.uint8 and np.max(gray) <= 1:
        bin_inv = (gray * 255).astype(np.uint8)
    elif len(np.unique(gray)) <= 2:
        bin_inv = gray.copy()
    else:
        bin_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 25, 8)

    # Horizontal morphological smearing
    k_w = max(15, int(w * 0.035))
    smeared = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 2)))
    smeared = cv2.morphologyEx(smeared, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(w * 0.02)), 1)))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(smeared)
    lines = []
    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        if cw > w * 0.08 and min_line_height <= ch <= int(h * 0.08):
            lines.append((y, y + ch))

    # Fallback to projection profile if fewer than 2 lines found
    if len(lines) < 2:
        proj = np.sum(bin_inv, axis=1).astype(np.float32)
        kernel = np.ones(11) / 11.0
        proj_smooth = np.convolve(proj, kernel, mode="same")
        threshold = max(proj_smooth.max() * 0.08, 1.0)
        in_line = proj_smooth > threshold
        start = None
        for i, active in enumerate(in_line):
            if active and start is None:
                start = i
            elif not active and start is not None:
                if i - start >= min_line_height:
                    lines.append((start, i))
                start = None
        if start is not None and len(bin_inv) - start >= min_line_height:
            lines.append((start, len(bin_inv)))

    # Sort and merge overlapping or very close lines
    lines.sort(key=lambda item: item[0])
    merged = []
    for s, e in lines:
        if merged and s <= merged[-1][1] + min_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    return merged


def auto_orient_document(img: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Automatically detect if document image needs 90 degree clockwise rotation
    to make text lines horizontal.
    
    Returns:
        (oriented_image, rotation_angle_applied)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    l0 = len(detect_text_lines(gray))
    
    img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    gray90 = cv2.cvtColor(img90, cv2.COLOR_BGR2GRAY) if len(img90.shape) == 3 else img90
    l90 = len(detect_text_lines(gray90))
    
    if l90 > l0 * 1.5 and l90 >= 4:
        return img90, 90
    return img, 0


# ══════════════════════════════════════════════════════════════════════════════
# 3. Coupled Snake Fitting (Active Contours per Text Line)
# ══════════════════════════════════════════════════════════════════════════════

def fit_coupled_snakes(img_gray: np.ndarray,
                       text_lines: list[tuple[int, int]],
                       n_control_pts: int = 25) -> list[dict]:
    """
    Fit coupled contours to each text line to accurately track baseline curvature.
    Fast, robust, and maintains spatial coherence without gradient divergence.

    Returns:
        List of dicts with 'y_range', 'upper_snake', 'lower_snake', 'midline'
    """
    h, w = img_gray.shape[:2]
    # Fast inverted adaptive threshold to isolate character ink strokes
    if len(np.unique(img_gray)) <= 2:
        bin_inv = img_gray.copy()
    else:
        bin_inv = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 8
        )

    xs = np.linspace(15, w - 15, n_control_pts, dtype=np.float32)
    results = []

    for (y_start, y_end) in text_lines:
        line_h = max(y_end - y_start, 6)
        pad = max(2, int(line_h * 0.2))
        y0 = max(0, y_start - pad)
        y1 = min(h, y_end + pad)
        strip = bin_inv[y0:y1, :]

        y_mids = []
        for x in xs:
            ix = int(np.clip(round(x), 0, w - 1))
            col_strip = strip[:, max(0, ix - 3):min(w, ix + 4)]
            ink_rows = np.where(col_strip > 0)[0]
            if len(ink_rows) >= 2:
                ymid = y0 + float(np.median(ink_rows))
            else:
                ymid = (y_start + y_end) / 2.0
            y_mids.append(ymid)

        y_mids_arr = np.array(y_mids, dtype=np.float32)
        # 2nd order polynomial fit for ultra smooth, noise-free midline
        try:
            poly2 = np.polyfit(xs, y_mids_arr, 2)
            y_smooth = np.polyval(poly2, xs)
        except Exception:
            y_smooth = y_mids_arr

        half_h = line_h / 2.0
        results.append({
            "y_range": (y_start, y_end),
            "upper_snake": np.column_stack([xs, y_smooth - half_h]),
            "lower_snake": np.column_stack([xs, y_smooth + half_h]),
            "midline": np.column_stack([xs, y_smooth])
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5. Curvature Estimation & Cylindrical Displacement Map
# ══════════════════════════════════════════════════════════════════════════════

def build_displacement_map(img_shape: tuple,
                           snake_results: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a pixel-level displacement map (map_x, map_y) from snake midlines.

    True Cylindrical Baseline Dewarping:
      - Extracts the pure horizontal curvature deviation delta(x) relative to linear trend.
      - Strictly preserves vertical distances between lines (d y_map / d y = 1.0).
      - Never stretches text vertically, never alters font height or paragraph layout.
      - If median curvature amplitude < 6.0 px, the document is planar and dewarping is bypassed.
      - When curvature is present, shifts columns vertically by delta(x) to flatten smiles/frowns.

    Args:
        img_shape:     (height, width[, channels]) of the image
        snake_results: Output of fit_coupled_snakes()

    Returns:
        (map_x, map_y) — float32 arrays of shape (H, W)
    """
    h, w = img_shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    if not snake_results or len(snake_results) < 2:
        return grid_x, grid_y

    curvatures = []
    amps = []
    xs = snake_results[0]["midline"][:, 0]

    for res in snake_results:
        pts = res["midline"]
        lx = pts[:, 0]
        ly = pts[:, 1]
        try:
            poly1 = np.polyfit(lx, ly, 1)
            trend = np.polyval(poly1, lx)
            curve = ly - trend
            amp = float(np.max(curve) - np.min(curve))
            # Ignore lines with wild outliers (e.g. crossing borders or margin artifacts)
            if amp < h * 0.12:
                curvatures.append(curve)
                amps.append(amp)
        except Exception:
            continue

    if not curvatures or len(curvatures) < 2:
        return grid_x, grid_y

    median_amp = float(np.median(amps))
    # Planar threshold: if page curl amplitude < 6.0 px, document is flat
    if median_amp < 6.0:
        return grid_x, grid_y

    # Robust median curvature profile across all lines
    med_curve = np.median(curvatures, axis=0)
    # Smooth curvature profile
    med_curve_smooth = cv2.GaussianBlur(med_curve.reshape(1, -1), (5, 1), 0).ravel()

    # Interpolate across full image width
    full_curve = np.interp(np.arange(w, dtype=np.float32), xs, med_curve_smooth)
    # Gaussian blur the 1D profile across columns for seamless transition
    k_smooth = max(31, int(w * 0.03) | 1)
    full_curve = cv2.GaussianBlur(full_curve.reshape(1, -1), (k_smooth, 1), 0).ravel()

    # Clamp max displacement to 10% of page height
    max_offset = h * 0.10
    full_curve = np.clip(full_curve, -max_offset, max_offset)

    # Pure vertical displacement map: d(map_y)/dy = 1.0 (Zero text stretching!)
    map_x = grid_x
    map_y = np.clip(grid_y + full_curve[np.newaxis, :], 0, h - 1).astype(np.float32)

    return map_x, map_y


# ══════════════════════════════════════════════════════════════════════════════
# 6. Master Dewarp Function
# ══════════════════════════════════════════════════════════════════════════════

def dewarp(img: np.ndarray, debug: bool = False, auto_orient: bool = True) -> tuple[np.ndarray, dict]:
    """
    Full dewarping pipeline for a perspective-corrected document image.

    - Checks and applies 90 degree auto-rotation if text lines run vertically.
    - Uses morphological smearing to reliably segment lines on lined/plain paper.
    - Fits coupled snakes to track upper/lower boundaries of text lines.
    - Estimates bivariate page curvature and applies Lanczos displacement remapping.

    Args:
        img:         Perspective-corrected BGR image
        debug:       If True, include intermediate visualizations in debug_info
        auto_orient: Automatically rotate 90 deg if text lines run vertically

    Returns:
        (dewarped_img, debug_info dict)
    """
    debug_info: dict = {}
    current = img

    if auto_orient:
        current, rot_angle = auto_orient_document(current)
        debug_info["auto_rotation_deg"] = rot_angle
    else:
        debug_info["auto_rotation_deg"] = 0

    if len(current.shape) == 3:
        gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    else:
        gray = current.copy()

    # ── Step 1: Binarize ─────────────────────────────────────────────────────
    binary = binarize(gray)
    debug_info["binary"] = binary

    # ── Step 2: Detect text lines ────────────────────────────────────────────
    text_lines = detect_text_lines(gray)
    debug_info["text_lines"] = text_lines
    debug_info["n_lines"] = len(text_lines)

    if len(text_lines) < 2:
        print("[DEWARP] < 2 text lines detected — skipping dewarping.")
        return current.copy(), debug_info

    # ── Step 3: Fit coupled snakes (sample up to 25 lines for speed) ──────────
    n_use = min(len(text_lines), 25)
    step = max(1, len(text_lines) // n_use)
    selected_lines = text_lines[::step][:n_use]

    snake_results = fit_coupled_snakes(gray, selected_lines, n_control_pts=20)
    debug_info["snake_results"] = snake_results

    # ── Step 4: Build displacement map ───────────────────────────────────────
    map_x, map_y = build_displacement_map(current.shape, snake_results)
    debug_info["map_x"] = map_x
    debug_info["map_y"] = map_y

    # Record actual max displacement for debug
    ref_y = np.arange(current.shape[0], dtype=np.float32)[:, None]
    max_delta = float(np.max(np.abs(map_y - ref_y)))
    debug_info["max_displacement_px"] = round(max_delta, 2)

    # ── Step 5: Remap image (bypass if document is already planar) ────────────
    if max_delta < 0.5:
        dewarped = current.copy()
    else:
        dewarped = cv2.remap(current, map_x, map_y,
                              interpolation=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_REPLICATE)

    if debug:
        debug_info["snake_overlay"] = _draw_snakes(current, snake_results)

    print(f"[DEWARP] {len(text_lines)} lines, {len(snake_results)} snakes, "
          f"max displacement = {max_delta:.1f} px")

    return dewarped, debug_info


def _draw_snakes(img: np.ndarray, snake_results: list[dict]) -> np.ndarray:
    """Visualize fitted snakes overlaid on the image."""
    out = img.copy()
    colors = {
        "upper": (0, 120, 255),   # Blue (BGR)
        "lower": (255, 80, 0),    # Orange (BGR)
        "mid":   (0, 220, 80),    # Green (BGR)
    }

    for res in snake_results:
        for key, color_key in [("upper_snake", "upper"),
                                ("lower_snake", "lower"),
                                ("midline",     "mid")]:
            pts = res[key].astype(np.int32)
            color = colors[color_key]
            for i in range(len(pts) - 1):
                p1 = (int(pts[i, 0]), int(pts[i, 1]))
                p2 = (int(pts[i + 1, 0]), int(pts[i + 1, 1]))
                cv2.line(out, p1, p2, color, 2)
            for pt in pts:
                cv2.circle(out, (int(pt[0]), int(pt[1])), 3, color, -1)

    return out
