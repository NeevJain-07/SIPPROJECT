"""
baseline.py — Homography-Only Baseline
Runs the pipeline WITHOUT the dewarping step (Phase 5 disabled).
Used to demonstrate the improvement our coupled-snake dewarping provides.

Usage:
    python baseline.py --image path/to/image.jpg
    python baseline.py --dataset dataset/warped/
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
from pipeline import run_pipeline


def run_baseline(source, output_dir: str = "outputs/baseline", verbose: bool = True):
    """
    Run baseline pipeline (no dewarping, homography only).

    Returns:
        PipelineResult
    """
    result = run_pipeline(
        source       = source,
        output_dir   = output_dir,
        save_outputs = True,
        run_dewarp   = False,    # ← Key difference from full pipeline
        run_enhance  = True,
        verbose      = verbose,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline: Homography-only correction")
    parser.add_argument("--image",   type=str, help="Single image path")
    parser.add_argument("--dataset", type=str, help="Directory of images")
    parser.add_argument("--output",  type=str, default="outputs/baseline")
    args = parser.parse_args()

    if args.image:
        result = run_baseline(args.image, args.output)
        print(f"\nBaseline Score: {result.quality_after.overall_score:.1f}/100")

    elif args.dataset:
        dataset_dir = Path(args.dataset)
        images = list(dataset_dir.glob("*.png")) + list(dataset_dir.glob("*.jpg"))
        print(f"[BASELINE] Running on {len(images)} images...")
        scores = []
        for img_path in images:
            r = run_baseline(str(img_path), args.output, verbose=False)
            if r.success and r.quality_after:
                scores.append(r.quality_after.overall_score)
                print(f"  {img_path.name}: {r.quality_after.overall_score:.1f}/100")
        if scores:
            print(f"\n[BASELINE] Mean score: {np.mean(scores):.1f}/100")
    else:
        parser.print_help()
