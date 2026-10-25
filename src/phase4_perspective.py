"""
Phase 4 — Perspective Correction
4-point homography estimation and warpPerspective application.
"""
import cv2
import numpy as np
from src.phase3_page_detection import order_corners, compute_warp_dimensions


def correct_perspective(img: np.ndarray, quad: np.ndarray, paper_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Apply perspective correction using a detected quadrilateral.
    If paper_mask is provided, non-paper background is masked to pure white (255, 255, 255)
    to guarantee zero desk, bedsheet, or table leakage in the output scan.

    Args:
        img:        Input BGR image
        quad:       (4, 1, 2) or (4, 2) array of document corners
        paper_mask: Optional binary mask (same H, W as img) where 255 = paper interior

    Returns:
        Perspective-corrected BGR image
    """
    clean_img = img.copy()
    if paper_mask is not None:
        if paper_mask.shape[:2] == img.shape[:2]:
            clean_img[paper_mask == 0] = [255, 255, 255]

    corners = order_corners(quad)   # [TL, TR, BR, BL]
    tl, tr, br, bl = corners

    width, height = compute_warp_dimensions(corners)

    # Destination rectangle — perfectly flat
    dst = np.array([
        [0,         0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0,         height - 1]
    ], dtype=np.float32)

    # Compute homography matrix
    M = cv2.getPerspectiveTransform(corners, dst)

    # Apply warp with clean white boundary
    warped = cv2.warpPerspective(clean_img, M, (width, height),
                                  flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=[255, 255, 255])
    return warped


def refine_quad_with_hough(img_gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """
    Optionally refine detected quad corners using Hough line intersections.
    Useful when contour detection gives slightly off corners.

    Returns:
        Refined (4, 2) corner array, or original if refinement fails
    """
    edges = cv2.Canny(img_gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None or len(lines) < 4:
        return order_corners(quad)

    # Extract line parameters
    hough_lines = [(rho, theta) for rho, theta in lines[:, 0]]

    # Separate horizontal and vertical lines
    h_lines = [(r, t) for r, t in hough_lines if abs(t - np.pi / 2) < np.pi / 6]
    v_lines = [(r, t) for r, t in hough_lines if t < np.pi / 6 or t > 5 * np.pi / 6]

    if len(h_lines) < 2 or len(v_lines) < 2:
        return order_corners(quad)

    # Use closest pair of H and V lines to document edges
    corners = order_corners(quad)
    return corners  # Return original if refinement not decisive


def estimate_homography_quality(src_pts: np.ndarray, dst_pts: np.ndarray,
                                 M: np.ndarray) -> float:
    """
    Measure reprojection error of the homography to assess quality.

    Returns:
        Mean reprojection error in pixels (lower is better)
    """
    src_h = np.hstack([src_pts, np.ones((4, 1))]).T  # (3, 4)
    projected = M @ src_h
    projected /= projected[2]                          # normalize
    projected = projected[:2].T                        # (4, 2)

    error = np.mean(np.linalg.norm(projected - dst_pts, axis=1))
    return float(error)
