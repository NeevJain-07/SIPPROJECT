"""
dataset_generator.py — Synthetic Warped Document Dataset Generator

Creates a dataset of realistically warped document images from flat originals.
Warp types:
  1. Perspective warp  (simulates camera angle / skew)
  2. Cylindrical warp  (simulates page curl)
  3. Gaussian surface  (arbitrary surface bend)
  4. Combo             (perspective + lighting + noise)

Usage:
    python dataset_generator.py --input dataset/raw --output dataset/warped --count 5
"""
import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import textwrap


# ══════════════════════════════════════════════════════════════════════════════
# Flat Document Synthesizer (if no real images provided)
# ══════════════════════════════════════════════════════════════════════════════

def create_synthetic_flat_document(idx: int = 0,
                                    width: int = 800,
                                    height: int = 1100) -> np.ndarray:
    """
    Generate a synthetic flat white document with printed text lines,
    headers, and a table — to simulate a real printed page.
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 245  # Off-white background

    # Page border
    cv2.rectangle(img, (30, 30), (width-30, height-30), (200, 200, 200), 2)

    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX]
    font = fonts[idx % 2]

    # Title
    cv2.putText(img, f"DOCUMENT TITLE - PAGE {idx+1}", (60, 80),
                font, 0.9, (20, 20, 80), 2)

    # Horizontal rule under title
    cv2.line(img, (60, 95), (width-60, 95), (100, 100, 100), 1)

    # Paragraph lines
    sample_lines = [
        "This is a sample document created for testing the dewarping pipeline.",
        "The document contains multiple lines of text at varying positions.",
        "Classical computer vision techniques are used for page detection.",
        "Perspective correction is applied using homography transformation.",
        "The coupled-snake algorithm models curvature of the page surface.",
        "Text lines are detected using horizontal projection profiles.",
        "Energy minimization finds optimal snake control point positions.",
        "A displacement map is computed and applied using cv2.remap().",
        "Image enhancement improves readability of the corrected output.",
        "Quality metrics include sharpness, blur, brightness, and contrast.",
        "The OCR readiness score predicts suitability for text extraction.",
        "An overall score out of 100 is computed with a star rating.",
        "Recommendations are generated based on detected quality issues.",
        "The system exports corrected images and PDF quality reports.",
        "Evaluation compares SSIM and PSNR against ground truth images.",
        "Runtime measurements are recorded for each processing phase.",
        "The baseline uses homography-only correction for comparison.",
        "This project demonstrates document image processing research.",
        "Future work includes deep learning integration for enhancement.",
        "The system supports multiple languages for OCR evaluation.",
    ]

    y_pos = 130
    for i, line in enumerate(sample_lines):
        # Vary font size slightly for realism
        scale = 0.42 + 0.02 * (i % 3)
        thickness = 1
        color = (10, 10, 30)
        cv2.putText(img, line, (60, y_pos), font, scale, color, thickness)
        y_pos += 28 + 4 * (i % 2)

        # Add paragraph break every 5 lines
        if (i + 1) % 5 == 0:
            y_pos += 15

    # Simple table
    table_y = y_pos + 20
    if table_y < height - 150:
        headers = ["Metric", "Before", "After", "Delta"]
        col_w, row_h = 150, 28
        for j, h_txt in enumerate(headers):
            x = 60 + j * col_w
            cv2.rectangle(img, (x, table_y), (x+col_w, table_y+row_h), (80, 80, 180), -1)
            cv2.putText(img, h_txt, (x+8, table_y+19), font, 0.45, (255,255,255), 1)

        rows = [("Sharpness", "62%", "91%", "+29%"),
                ("OCR Score", "71%", "94%", "+23%"),
                ("SSIM",      "0.71", "0.94", "+0.23")]
        for ri, row in enumerate(rows):
            ry = table_y + row_h * (ri+1)
            bg = (235, 235, 248) if ri % 2 == 0 else (245, 245, 255)
            cv2.rectangle(img, (60, ry), (60+col_w*4, ry+row_h), bg, -1)
            cv2.rectangle(img, (60, ry), (60+col_w*4, ry+row_h), (180,180,200), 1)
            for j, cell in enumerate(row):
                cv2.putText(img, cell, (60 + j*col_w + 8, ry+19),
                           font, 0.42, (30, 30, 80), 1)

    # Page number
    cv2.putText(img, f"— {idx+1} —", (width//2 - 20, height-40),
               font, 0.4, (120, 120, 120), 1)

    return img


# ══════════════════════════════════════════════════════════════════════════════
# Warp Functions
# ══════════════════════════════════════════════════════════════════════════════

def warp_perspective_random(img: np.ndarray,
                              jitter_pct: float = 0.15) -> tuple[np.ndarray, dict]:
    """
    Apply random perspective warp by jittering the four corners.

    Args:
        img:        Input image
        jitter_pct: Max corner displacement as fraction of image dimension

    Returns:
        (warped_img, warp_params)
    """
    h, w = img.shape[:2]
    jx = int(w * jitter_pct)
    jy = int(h * jitter_pct)

    rng = np.random.default_rng()

    def rj(lo, hi): return int(rng.integers(lo, hi))

    src = np.array([
        [rj(0,  jx),       rj(0,  jy)],        # TL
        [rj(w-jx, w),      rj(0,  jy)],        # TR
        [rj(w-jx, w),      rj(h-jy, h)],       # BR
        [rj(0,  jx),       rj(h-jy, h)],       # BL
    ], dtype=np.float32)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h))

    return warped, {"type": "perspective", "src_pts": src.tolist()}


def warp_cylindrical(img: np.ndarray,
                      strength: float = 0.3) -> tuple[np.ndarray, dict]:
    """
    Apply cylindrical (page curl) warp.
    Simulates a document page curling inward at left/right edges.

    Args:
        img:      Input image
        strength: Curl strength (0 = none, 1 = extreme)
    """
    h, w = img.shape[:2]
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    cx = w / 2.0
    f  = w / (2.0 * np.pi * strength) if strength > 0 else 1e9

    for y in range(h):
        for x in range(w):
            # Cylinder formula: map pixel x through arcsin
            rel = (x - cx) / (cx + 1e-6)
            theta = rel * np.pi * strength
            src_x = cx + f * np.sin(theta)
            src_y = y + (h * 0.05 * strength) * (1 - np.cos(theta))
            map_x[y, x] = np.clip(src_x, 0, w - 1)
            map_y[y, x] = np.clip(src_y, 0, h - 1)

    warped = cv2.remap(img, map_x, map_y,
                        cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    return warped, {"type": "cylindrical", "strength": strength}


def warp_gaussian_surface(img: np.ndarray,
                           amplitude: float = 30.0,
                           sigma: float = 0.4) -> tuple[np.ndarray, dict]:
    """
    Apply a Gaussian bump warp (arbitrary surface bend).
    Models a document lying on a curved surface.

    Args:
        amplitude: Max pixel displacement
        sigma:     Spatial spread (fraction of image size)
    """
    h, w = img.shape[:2]
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    cx, cy = w * 0.5, h * 0.5
    sx, sy = w * sigma, h * sigma

    # Gaussian bump: displacement in y direction
    bump = amplitude * np.exp(
        -((x_coords - cx)**2 / (2*sx**2) + (y_coords - cy)**2 / (2*sy**2))
    )

    map_x = x_coords
    map_y = np.clip(y_coords - bump, 0, h - 1)

    warped = cv2.remap(img, map_x, map_y,
                        cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    return warped, {"type": "gaussian_surface", "amplitude": amplitude, "sigma": sigma}


def add_lighting_variation(img: np.ndarray) -> np.ndarray:
    """Add realistic gradient lighting (simulates uneven room lighting)."""
    h, w = img.shape[:2]
    rng = np.random.default_rng()

    # Random gradient direction and strength
    direction = rng.choice(["left", "right", "top", "bottom"])
    strength  = rng.uniform(0.2, 0.5)

    gradient = np.ones((h, w), dtype=np.float32)
    if direction == "left":
        gradient = np.tile(np.linspace(1 - strength, 1, w), (h, 1))
    elif direction == "right":
        gradient = np.tile(np.linspace(1, 1 - strength, w), (h, 1))
    elif direction == "top":
        gradient = np.tile(np.linspace(1 - strength, 1, h).reshape(-1, 1), (1, w))
    else:
        gradient = np.tile(np.linspace(1, 1 - strength, h).reshape(-1, 1), (1, w))

    if len(img.shape) == 3:
        gradient = gradient[:, :, np.newaxis]

    lit = np.clip(img.astype(np.float32) * gradient, 0, 255).astype(np.uint8)
    return lit


def add_camera_noise(img: np.ndarray,
                      noise_std: float = 8.0) -> np.ndarray:
    """Add Gaussian camera noise."""
    noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


# ══════════════════════════════════════════════════════════════════════════════
# Dataset Generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_dataset(input_dir:  str | Path,
                      output_dir: str | Path,
                      gt_dir:     str | Path,
                      count_per_image: int = 5,
                      use_synthetic: bool = False) -> dict:
    """
    Generate a warped document dataset.

    Args:
        input_dir:       Directory of flat "raw" document images
        output_dir:      Where to save warped images
        gt_dir:          Where to copy ground truth (flat) images
        count_per_image: Number of warp variants per input image
        use_synthetic:   Generate synthetic flat docs if no real images exist

    Returns:
        Dataset manifest dict
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    gt_dir     = Path(gt_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Find input images
    exts = ["*.jpg", "*.jpeg", "*.png", "*.tiff", "*.bmp"]
    raw_images = []
    for ext in exts:
        raw_images.extend(sorted(input_dir.glob(ext)))

    if not raw_images:
        if use_synthetic:
            print(f"[DATASET] No real images found. Generating {count_per_image * 5} synthetic documents...")
            n_synth = max(10, count_per_image * 5)
            for i in range(n_synth):
                synth = create_synthetic_flat_document(i)
                p = input_dir / f"synth_{i:03d}.png"
                cv2.imwrite(str(p), synth)
                raw_images.append(p)
            print(f"[DATASET] Created {len(raw_images)} synthetic flat documents.")
        else:
            print("[DATASET] No images found. Run with --synthetic flag or add images to dataset/raw/")
            return {}

    warp_configs = [
        ("perspective_mild",   lambda img: warp_perspective_random(img, 0.08)),
        ("perspective_strong", lambda img: warp_perspective_random(img, 0.15)),
        ("cylindrical_mild",   lambda img: warp_cylindrical(img, 0.15)),
        ("cylindrical_strong", lambda img: warp_cylindrical(img, 0.30)),
        ("gaussian_bump",      lambda img: warp_gaussian_surface(img, 25, 0.4)),
    ]

    manifest = {"images": [], "total": 0}

    for raw_path in tqdm(raw_images, desc="Processing raw images"):
        img = cv2.imread(str(raw_path))
        if img is None:
            print(f"[DATASET] Cannot read: {raw_path}")
            continue

        # Resize to standard size for consistency
        img = cv2.resize(img, (800, 1100), interpolation=cv2.INTER_LANCZOS4)

        # Save ground truth
        gt_name = f"gt_{raw_path.stem}.png"
        gt_path = gt_dir / gt_name
        cv2.imwrite(str(gt_path), img)

        for warp_name, warp_fn in warp_configs[:count_per_image]:
            # Apply warp
            warped, warp_params = warp_fn(img)

            # Add lighting + noise
            warped = add_lighting_variation(warped)
            warped = add_camera_noise(warped, noise_std=np.random.uniform(3, 12))

            # Save warped image
            out_name = f"{raw_path.stem}_{warp_name}.png"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), warped)

            manifest["images"].append({
                "warped":       str(out_path),
                "ground_truth": str(gt_path),
                "warp_type":    warp_name,
                "warp_params":  warp_params,
                "source":       str(raw_path),
            })
            manifest["total"] += 1

    # Save manifest
    manifest_path = output_dir.parent / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[DATASET] Saved {manifest['total']} warped images.")
    print(f"[DATASET] Manifest saved -> {manifest_path}")
    return manifest


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthetic Warped Document Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Use real images from dataset/raw/
          python dataset_generator.py

          # Generate fully synthetic dataset
          python dataset_generator.py --synthetic

          # Custom directories and count
          python dataset_generator.py --input my_docs/ --count 3 --synthetic
        """)
    )
    parser.add_argument("--input",     default="dataset/raw",          help="Input flat image directory")
    parser.add_argument("--output",    default="dataset/warped",        help="Output warped image directory")
    parser.add_argument("--gt",        default="dataset/ground_truth",  help="Ground truth directory")
    parser.add_argument("--count",     type=int, default=5,             help="Warp variants per image (max 5)")
    parser.add_argument("--synthetic", action="store_true",             help="Generate synthetic flat documents if no real images")
    args = parser.parse_args()

    generate_dataset(
        input_dir        = args.input,
        output_dir       = args.output,
        gt_dir           = args.gt,
        count_per_image  = args.count,
        use_synthetic    = args.synthetic,
    )
