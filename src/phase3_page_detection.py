"""
Phase 3 — Robust Multi-Strategy Page/Paper Detection
Optimised for real-world photographs: bright paper on coloured/textured
backgrounds (desks, floors, tables).

Detection pipeline:
  A. Bright-region segmentation  (HSV V-channel + low saturation mask)
  B. Adaptive edge + contour     (Canny across LAB L-channel)
  C. GrabCut-seeded segmentation (for high-clutter scenes)
  D. Morphological flood-fill    (fallback for low-contrast scenes)

Each candidate quadrilateral is scored on:
  • Interior mean lightness vs exterior  (paper is bright)
  • Interior mean saturation             (white paper = low sat)
  • Convexity ratio                      (paper edge is convex)
  • Aspect ratio plausibility            (A4/letter proportions)
  • 4-sidedness of the approximation

The highest-scoring quad is selected, corners are ordered [TL,TR,BR,BL]
and optionally refined with cornerSubPix.
"""
from __future__ import annotations

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def detect_page_quad(
    img: np.ndarray,
    draw: bool = True,
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Detect the document's four corners in *img*.

    Parameters
    ----------
    img : BGR input image (any resolution).
    draw: If True the returned annotated image has the quad overlaid.

    Returns
    -------
    (quad, annotated_img)
    quad : (4, 1, 2) int32 contour in *original* image coordinates, or None.
    """
    if img is None or img.size == 0:
        return None, img if img is not None else np.zeros((10, 10, 3), dtype=np.uint8)

    h_orig, w_orig = img.shape[:2]

    # ── Step 1 : Downscale for speed ─────────────────────────────────────────
    MAX_DIM = 1024
    scale = 1.0
    if max(h_orig, w_orig) > MAX_DIM:
        scale = MAX_DIM / float(max(h_orig, w_orig))
        work = cv2.resize(img, (int(w_orig * scale), int(h_orig * scale)),
                          interpolation=cv2.INTER_AREA)
    else:
        work = img.copy()

    h, w = work.shape[:2]
    work_area = h * w

    # ── Step 2 : Colour-space decomposition ──────────────────────────────────
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    lab  = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)

    H_ch = hsv[:, :, 0].astype(np.float32)
    S_ch = hsv[:, :, 1].astype(np.float32)
    V_ch = hsv[:, :, 2].astype(np.float32)
    L_ch = lab[:, :, 0].astype(np.float32)

    # ── Step 3 : Build candidate binary masks ────────────────────────────────
    masks: list[np.ndarray] = []

    # --- Mask A : bright low-saturation region (white / light-coloured paper) --
    # V > 160 AND S < 80  =>  bright and nearly achromatic
    mask_bright = cv2.inRange(
        hsv,
        np.array([0,   0, 150], dtype=np.uint8),
        np.array([180, 90, 255], dtype=np.uint8),
    )
    masks.append(_clean_mask(mask_bright, ksize=15))

    # --- Mask B : Otsu on L-channel -------------------------------------------
    _, otsu_L = cv2.threshold(
        L_ch.astype(np.uint8), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    masks.append(_clean_mask(otsu_L, ksize=13))

    # --- Mask C : Adaptive threshold on L-channel (handles uneven lighting) ---
    adapt = cv2.adaptiveThreshold(
        L_ch.astype(np.uint8), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 8,
    )
    masks.append(_clean_mask(adapt, ksize=11))

    # --- Mask D : Canny edges → dilate → close → flood-fill ------------------
    blurred_gray = cv2.bilateralFilter(gray, 9, 80, 80)
    # Auto Canny thresholds via Otsu
    otsu_val, _ = cv2.threshold(blurred_gray, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(blurred_gray,
                      max(10.0, otsu_val * 0.3),
                      min(255.0, otsu_val * 1.0))
    # Dilate edges then close to form solid regions
    edge_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_edges = cv2.dilate(edges, edge_k, iterations=2)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, close_k)
    # Flood-fill from border to separate foreground from background
    ff = closed_edges.copy()
    mask_ff = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(ff, mask_ff, (0, 0), 255)
    ff_inv = cv2.bitwise_not(ff)
    masks.append(_clean_mask(cv2.bitwise_or(closed_edges, ff_inv), ksize=9))

    # --- Mask E : Saturation-inverted mask (paper = low sat) ------------------
    # Low saturation areas (S < 60) thresholded
    _, low_sat = cv2.threshold(
        S_ch.astype(np.uint8), 60, 255, cv2.THRESH_BINARY_INV
    )
    bright_enough = cv2.inRange(V_ch.astype(np.uint8), 120, 255)
    mask_sat = cv2.bitwise_and(low_sat, bright_enough)
    masks.append(_clean_mask(mask_sat, ksize=17))

    # --- Mask F : Seeded GrabCut segmentation (for patterned backgrounds) ---
    try:
        gc_scale = 500.0 / max(h, w) if max(h, w) > 500 else 1.0
        gc_work = cv2.resize(work, (int(w * gc_scale), int(h * gc_scale))) if gc_scale != 1.0 else work.copy()
        gh, gw = gc_work.shape[:2]
        gc_mask = np.zeros((gh, gw), np.uint8)
        rect = (int(gw * 0.03), int(gh * 0.03), int(gw * 0.94), int(gh * 0.94))
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(gc_work, gc_mask, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
        gc_fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        if gc_scale != 1.0:
            gc_fg = cv2.resize(gc_fg, (w, h), interpolation=cv2.INTER_NEAREST)
        masks.append(_clean_mask(gc_fg, ksize=15))
    except Exception:
        pass

    # ── Step 4 : Extract quad candidates from every mask ─────────────────────
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []

    for mask in masks:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            cnt_area = cv2.contourArea(cnt)
            ar = cnt_area / work_area
            if ar < 0.04 or ar > 0.999:      # too small or entire frame
                continue
            _extract_quads_from_contour(cnt, work, L_ch, S_ch,
                                         work_area, candidates, source_mask=mask)

    # ── Step 5 : Select best quad ─────────────────────────────────────────────
    best_quad_orig: np.ndarray | None = None
    annotated = img.copy()
    paper_mask_orig: np.ndarray | None = None

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_q_work, src_m = candidates[0]

        if best_score > 10.0:           # quality threshold
            ordered_work = order_corners(best_q_work)

            # Refine corners with sub-pixel accuracy
            gray_work = gray.copy()
            refined = _refine_corners(gray_work, ordered_work)

            # Scale back to original resolution
            ordered_orig = (refined / scale).astype(np.float32) \
                           if scale != 1.0 else refined.astype(np.float32)

            best_quad_orig = ordered_orig.reshape(4, 1, 2).astype(np.int32)
            
            # Build high-res paper mask with safety inset to eliminate edge leakage
            if src_m is not None:
                m_full = cv2.resize(src_m, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                # Inset by ~1.5%
                k_inset = max(5, int(min(w_orig, h_orig) * 0.015) | 1)
                paper_mask_orig = cv2.erode(m_full, cv2.getStructuringElement(cv2.MORPH_RECT, (k_inset, k_inset)))
            else:
                paper_mask_orig = np.zeros((h_orig, w_orig), dtype=np.uint8)
                cv2.fillConvexPoly(paper_mask_orig, best_quad_orig, 255)

    # ── Step 6 : Fallback ─────────────────────────────────────────────────────
    if best_quad_orig is None:
        best_quad_orig = _fallback_quad(img)
        paper_mask_orig = np.zeros((h_orig, w_orig), dtype=np.uint8)
        cv2.fillConvexPoly(paper_mask_orig, best_quad_orig, 255)

    # ── Step 7 : Draw ─────────────────────────────────────────────────────────
    if draw and best_quad_orig is not None:
        _draw_quad(annotated, best_quad_orig)

    return best_quad_orig, annotated, paper_mask_orig


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_mask(mask: np.ndarray, ksize: int = 11) -> np.ndarray:
    """Morphological close + open to remove holes and noise specks."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    return opened


def _extract_quads_from_contour(
    cnt: np.ndarray,
    work_img: np.ndarray,
    L_ch: np.ndarray,
    S_ch: np.ndarray,
    work_area: float,
    candidates: list[tuple[float, np.ndarray, np.ndarray]],
    source_mask: np.ndarray | None = None,
) -> None:
    """Try multiple approximation strategies on one contour and score each."""
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    hull_area = cv2.contourArea(hull)
    cnt_area  = cv2.contourArea(cnt)

    # Convexity ratio (paper boundary should be convex)
    convexity = hull_area / (cnt_area + 1e-6)

    quad_list: list[np.ndarray] = []

    # Method A : Dynamic epsilon poly-approx (seek exactly 4 sides)
    for eps in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad_list.append(approx.reshape(4, 2).astype(np.float32))

    # Method B : Directional-projection 4-corner extraction
    hull_pts = hull.reshape(-1, 2).astype(np.float32)
    if len(hull_pts) >= 4:
        s    = hull_pts.sum(axis=1)
        diff = hull_pts[:, 1] - hull_pts[:, 0]
        tl = hull_pts[np.argmin(s)]
        br = hull_pts[np.argmax(s)]
        tr = hull_pts[np.argmin(diff)]
        bl = hull_pts[np.argmax(diff)]
        proj = np.array([tl, tr, br, bl], dtype=np.float32)
        if cv2.isContourConvex(proj.astype(np.int32)):
            quad_list.append(proj)

    # Method C : Minimum-area bounding rect
    rect = cv2.minAreaRect(hull)
    box  = cv2.boxPoints(rect).astype(np.float32)
    quad_list.append(box)

    # Score each
    for q in quad_list:
        s = _score_quad(q, work_img, L_ch, S_ch, work_area, convexity)
        if s > -50:
            candidates.append((s, q.copy(), source_mask))


def _score_quad(
    q_pts: np.ndarray,
    img: np.ndarray,
    L_ch: np.ndarray,
    S_ch: np.ndarray,
    work_area: float,
    convexity: float,
) -> float:
    """
    Score a candidate quad.  Higher = better document candidate.

    Weights:
      Lightness contrast (inside vs outside)  — strongest signal
      Interior low-saturation                 — white paper has low sat
      Convexity                               — paper is a convex shape
      Area ratio                              — avoid extremes
      Aspect ratio                            — A4/letter proportions
    """
    q_int = q_pts.astype(np.int32)
    area  = cv2.contourArea(q_int)
    if area <= 0:
        return -100.0

    ar = area / work_area
    if ar < 0.04 or ar > 0.999:
        return -50.0

    h_img, w_img = img.shape[:2]

    # Build inside / outside masks
    mask_in  = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.fillConvexPoly(mask_in, q_int, 255)
    mask_out = cv2.bitwise_not(mask_in)

    # Lightness statistics
    mean_L_in  = float(cv2.mean(L_ch, mask=mask_in)[0])
    mean_L_out = float(cv2.mean(L_ch, mask=mask_out)[0])
    
    out_area = (h_img * w_img) - area
    if out_area < (h_img * w_img) * 0.05:
        # If the background is too small (<5% of image), contrast is unreliable
        l_contrast = 0.0
    else:
        l_contrast = mean_L_in - mean_L_out     # positive = paper brighter than bg

    # Saturation inside (white paper → low saturation)
    mean_S_in = float(cv2.mean(S_ch, mask=mask_in)[0])

    # Aspect ratio score (prefer A-paper / landscape/portrait ratios 1.1–2.5)
    rect = cv2.minAreaRect(q_int)
    rw, rh = rect[1]
    aspect_score = 0.0
    if rw > 0 and rh > 0:
        ratio = max(rw, rh) / (min(rw, rh) + 1e-6)
        if 1.05 <= ratio <= 2.8:
            aspect_score = 15.0
        elif ratio <= 1.05:           # nearly square — less likely to be paper
            aspect_score = -5.0

    # Area bonus: prefer 15%–80% of frame
    area_bonus = 0.0
    if 0.15 <= ar <= 0.80:
        area_bonus = 10.0
    elif 0.05 <= ar < 0.15 or 0.80 < ar <= 0.98:
        area_bonus = 2.0

    # Border penalty: heavily penalize if quad corners touch image edges
    margin = 5
    touches_border = False
    for pt in q_int:
        x, y = pt[0], pt[1]
        if x <= margin or x >= w_img - margin or y <= margin or y >= h_img - margin:
            touches_border = True
            
    border_penalty = -40.0 if touches_border else 0.0

    # Rectangularity bonus — penalize quads with heavily cut/skewed non-perpendicular corners
    ordered_q = order_corners(q_int)
    # Check angles at 4 corners
    v1 = ordered_q[1] - ordered_q[0] # TL->TR
    v2 = ordered_q[3] - ordered_q[0] # TL->BL
    cos_tl = abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
    ortho_bonus = (1.0 - cos_tl) * 15.0 # Higher bonus when corners are ~90 degrees

    # Composite — lightness contrast is the dominant term
    score = (
        l_contrast   * 1.8   +    # bright interior vs dark bg
        mean_L_in    * 0.3   +    # absolute brightness (paper tends bright)
        (100.0 - mean_S_in) * 0.5 +   # low saturation bonus
        convexity    * 12.0  +    # convex hull bonus
        ortho_bonus          +    # right-angle corner bonus (prevents diagonal cuts across folds)
        area_bonus            +
        aspect_score          +
        border_penalty
    )
    return float(score)


def _refine_corners(
    gray: np.ndarray,
    corners: np.ndarray,  # (4, 2) float32
    win: int = 11,
) -> np.ndarray:
    """
    Sub-pixel corner refinement using cornerSubPix.
    Falls back to original corners if refinement fails.
    """
    try:
        pts = corners.reshape(-1, 1, 2).astype(np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            40, 0.001,
        )
        refined = cv2.cornerSubPix(
            gray, pts,
            (win, win), (-1, -1),
            criteria,
        )
        return refined.reshape(4, 2)
    except Exception:
        return corners


def _fallback_quad(img: np.ndarray) -> np.ndarray:
    """Safe fallback: 90 % centred frame."""
    h, w = img.shape[:2]
    mw, mh = int(w * 0.05), int(h * 0.05)
    return np.array([
        [[mw,       mh]],
        [[w - mw,   mh]],
        [[w - mw,   h - mh]],
        [[mw,       h - mh]],
    ], dtype=np.int32)


def _draw_quad(img: np.ndarray, quad: np.ndarray) -> None:
    """Draw detected quad with labelled corners on *img* in-place."""
    pts = quad.reshape(4, 2)
    ordered = order_corners(pts)

    # Outline
    cv2.polylines(img, [ordered.astype(np.int32)], True, (0, 230, 80), 4)

    # Coloured corner dots + labels
    corner_colors = [
        (255,  60,  60),   # TL — red
        ( 60, 255,  60),   # TR — green
        ( 60,  60, 255),   # BR — blue
        (255, 255,   0),   # BL — yellow
    ]
    labels = ["TL", "TR", "BR", "BL"]
    for pt, color, label in zip(ordered, corner_colors, labels):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img, (x, y), 14, color, -1)
        cv2.circle(img, (x, y), 14, (255, 255, 255), 2)
        cv2.putText(img, label, (x + 16, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry utilities  (used by phase4_perspective.py)
# ─────────────────────────────────────────────────────────────────────────────

def order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 corner points as [TL, TR, BR, BL].

    Parameters
    ----------
    pts : (4, 2) or (4, 1, 2) array.

    Returns
    -------
    (4, 2) float32 ordered array.
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s    = pts.sum(axis=1)
    diff = pts[:, 1] - pts[:, 0]

    rect[0] = pts[np.argmin(s)]       # TL: smallest x+y
    rect[2] = pts[np.argmax(s)]       # BR: largest  x+y
    rect[1] = pts[np.argmin(diff)]    # TR: smallest y-x
    rect[3] = pts[np.argmax(diff)]    # BL: largest  y-x

    return rect


def compute_warp_dimensions(corners: np.ndarray) -> tuple[int, int]:
    """
    Compute output width and height from ordered corners [TL,TR,BR,BL].

    Uses the maximum of opposite edge lengths to preserve proportions.

    Returns
    -------
    (width, height) in pixels.
    """
    tl, tr, br, bl = corners

    top_w  = float(np.linalg.norm(tr - tl))
    bot_w  = float(np.linalg.norm(br - bl))
    left_h = float(np.linalg.norm(bl - tl))
    right_h= float(np.linalg.norm(br - tr))

    width  = max(int(max(top_w,  bot_w)),  1)
    height = max(int(max(left_h, right_h)), 1)

    return width, height
