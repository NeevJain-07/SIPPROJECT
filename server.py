"""
server.py — Clean, Lightweight Web Application Backend
Serves the interactive document scanner web interface and REST API.
"""
import io
import os
import glob
import base64
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory

from pipeline import run_pipeline, PipelineResult
from src.phase1_input import load_image

app = Flask(__name__, static_folder="static", static_url_path="")
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "web_runs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cache for latest processed result for download
LATEST_RESULT = {
    "image_path": None,
    "pdf_path": None,
    "filename": "scanned_document.png"
}


def to_base64_jpeg(img: np.ndarray, max_dim: int = 1400, quality: int = 88) -> str:
    """Encode numpy image as base64 JPEG data URL for display."""
    if img is None:
        return ""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def to_base64_png(img: np.ndarray, max_dim: int = 1400) -> str:
    """Encode numpy image as base64 PNG data URL for lossless display."""
    if img is None:
        return ""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode(".png", img)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/samples")
def get_samples():
    """List available sample images."""
    sample_files = sorted(glob.glob(str(PROJECT_ROOT / "IMG_*.jpeg")))
    samples = []
    descriptions = {
        "IMG_3712.jpeg": "Handwritten Notes on Table (Skewed + Fold)",
        "IMG_3715.jpeg": "Lined Notebook on Bedsheet (Creased)",
        "IMG_3716.jpeg": "Notes with Heavy Diagonal Shadow",
        "IMG_3717.jpeg": "Lined Spiral Notepad on Bedsheet",
        "IMG_3718.jpeg": "Heavily Crumpled / Wrinkled Paper",
    }
    for p in sample_files:
        name = Path(p).name
        samples.append({
            "name": name,
            "title": name,
            "description": descriptions.get(name, "Document Image"),
            "path": f"/api/thumbnail/{name}"
        })
    return jsonify({"samples": samples})


@app.route("/api/thumbnail/<filename>")
def get_thumbnail(filename):
    """Generate or serve cached fast thumbnail."""
    path = PROJECT_ROOT / filename
    if not path.exists():
        return "Not found", 404
    img = load_image(str(path))
    h, w = img.shape[:2]
    scale = 320.0 / max(h, w)
    thumb = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return send_file(io.BytesIO(buffer), mimetype="image/jpeg")


@app.route("/api/process", methods=["POST"])
def process_document():
    """Process an uploaded document image or selected sample."""
    try:
        rotation = int(request.form.get("rotation", 0))
        scan_mode = request.form.get("scan_mode", "clean_color")
        trim_margin = float(request.form.get("trim_margin", 0.015))
        run_dewarp = request.form.get("dewarp", "true").lower() == "true"
        sample_name = request.form.get("sample_name", "")

        img_input = None
        source_label = "uploaded_image"

        if "file" in request.files and request.files["file"].filename:
            file = request.files["file"]
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img_input = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            source_label = Path(file.filename).stem
        elif sample_name:
            sample_path = PROJECT_ROOT / sample_name
            if sample_path.exists():
                img_input = load_image(str(sample_path), rotation=0)
                source_label = Path(sample_name).stem

        if img_input is None:
            return jsonify({"success": False, "error": "No valid image provided."}), 400

        # Execute full pipeline
        res: PipelineResult = run_pipeline(
            source=img_input,
            output_dir=OUTPUT_DIR,
            save_outputs=True,
            run_dewarp=run_dewarp,
            run_enhance=True,
            rotation=rotation,
            scan_mode=scan_mode,
            trim_margin=trim_margin,
            verbose=False
        )

        if not res.success:
            return jsonify({"success": False, "error": res.error_message}), 500

        # Cache file paths for download
        LATEST_RESULT["image_path"] = str(res.saved_image_path) if res.saved_image_path else None
        LATEST_RESULT["pdf_path"] = str(res.saved_pdf_path) if res.saved_pdf_path else None
        LATEST_RESULT["filename"] = f"{source_label}_scan.png"

        # Determine scanned image to display
        def apply_trim(img, margin):
            if img is None:
                return None
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if margin > 0:
                sh, sw = img.shape[:2]
                ty = int(sh * margin)
                tx = int(sw * margin)
                if sh - 2*ty > 50 and sw - 2*tx > 50:
                    return img[ty:sh-ty, tx:sw-tx]
            return img

        color_img = res.enhanced.get("clean_color", res.dewarped)
        bw_img = res.enhanced.get("clean_bw", res.dewarped)
        photo_img = res.dewarped if res.dewarped is not None else res.perspective_corrected

        clean_color_trim = apply_trim(color_img, trim_margin)
        clean_bw_trim = apply_trim(bw_img, trim_margin)
        photo_trim = apply_trim(photo_img, trim_margin)

        mode_dict = {
            "clean_color": clean_color_trim,
            "clean_bw": clean_bw_trim,
            "dewarped_photo": photo_trim
        }
        scanned_img = mode_dict.get(scan_mode, clean_color_trim)

        # Extract quality metrics
        q_before = res.quality_before.to_dict() if res.quality_before else {}
        q_after = res.quality_after.to_dict() if res.quality_after else {}

        return jsonify({
            "success": True,
            "page_detected": res.page_detected,
            "total_time_sec": round(res.total_time, 2),
            "timings_ms": {k: round(v * 1000, 1) for k, v in res.timings.items()},
            "quality_before": {
                "overall_score": round(q_before.get("overall_score", 0), 1),
                "sharpness": round(q_before.get("sharpness_score", 0), 1),
                "blur_level": q_before.get("blur_level", "Medium"),
                "contrast": round(q_before.get("contrast_rms", 0), 1),
                "ocr_readiness": round(q_before.get("ocr_readiness", 0), 1),
            },
            "quality_after": {
                "overall_score": round(q_after.get("overall_score", 0), 1),
                "sharpness": round(q_after.get("sharpness_score", 0), 1),
                "blur_level": q_after.get("blur_level", "Low"),
                "contrast": round(q_after.get("contrast_rms", 0), 1),
                "ocr_readiness": round(q_after.get("ocr_readiness", 0), 1),
            },
            "images": {
                "original": to_base64_jpeg(res.original, max_dim=1200),
                "quad_overlay": to_base64_jpeg(res.quad_overlay, max_dim=1200),
                "dewarped": to_base64_jpeg(res.dewarped, max_dim=1400) if res.dewarped is not None else "",
                "scanned": to_base64_png(scanned_img, max_dim=1600),
                "modes": {
                    "clean_color": to_base64_png(clean_color_trim, max_dim=1600),
                    "clean_bw": to_base64_png(clean_bw_trim, max_dim=1600),
                    "dewarped_photo": to_base64_png(photo_trim, max_dim=1600)
                }
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/download/<fmt>")
def download_output(fmt):
    """Download scanned result as PNG or PDF report."""
    if fmt == "pdf" and LATEST_RESULT["pdf_path"] and os.path.exists(LATEST_RESULT["pdf_path"]):
        return send_file(LATEST_RESULT["pdf_path"], as_attachment=True, download_name="scan_report.pdf")
    elif LATEST_RESULT["image_path"] and os.path.exists(LATEST_RESULT["image_path"]):
        return send_file(LATEST_RESULT["image_path"], as_attachment=True, download_name=LATEST_RESULT["filename"])
    return "No file available to download", 404


if __name__ == "__main__":
    import webbrowser
    port = 5000
    url = f"http://localhost:{port}"
    print(f"\n=======================================================")
    print(f"  Document Dewarping & Scanning Web App")
    print(f"  Running at: {url}")
    print(f"=======================================================\n")
    # Automatically open browser
    try:
        webbrowser.open(url)
    except Exception:
        pass
    app.run(host="0.0.0.0", port=port, debug=False)
