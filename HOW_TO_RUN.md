# Project Submission & Execution Guide
## Document Dewarping & Scanning System
**Real-Time Classical Computer Vision Pipeline & Scan Quality Assessment**

---

### 1. Executive Summary
This project delivers a complete, end-to-end, high-precision document dewarping and scanning pipeline built strictly with classical computer vision (**OpenCV, NumPy, SciPy**):
- **100% Background Elimination**: Seeded GrabCut graph cuts combined with inward safety-margin mask erosion ($M_{safe} = M_{paper} \ominus K_\delta$) and pure white constant padding guarantee zero non-paper pixels leak into the output.
- **Physical Cylindrical Rectification**: Coupled active contours with developable cylindrical uncurling preserving $\frac{\partial y_{\text{map}}}{\partial y} \equiv 1.0$ and automated planar bypass ($< 6.0\text{ px}$ curvature threshold), eliminating artificial funhouse vertical stretching.
- **CamScanner-Grade Illumination Flat-Fielding**: Multi-scale morphological division ($I / B \times 255$) and contrast normalization neutralizing shadows and page yellowing.
- **Dual Production User Interfaces**:
  1. **Interactive Web Studio** (`run_app.bat` / `server.py` at `http://localhost:5000`): Real-time before/after split slider, instant mode switching (*Clean Color*, *Clean B&W*, *Dewarped Photo*), drag-and-drop custom upload, and PDF/PNG export.
  2. **Streamlit Diagnostic & Analytics Studio** (`launch_app.bat` / `app.py` at `http://127.0.0.1:8501`): Radial gauge indicators, intermediate pipeline stages debug view, batch evaluation, and research grounding.

---

### 2. Submission Files Directory

| File / Folder | Purpose & Content |
| :--- | :--- |
| **`REPORT.docx`** | **Complete Technical Project Report in Microsoft Word format** (37+ MB with embedded real screenshots, 6-paper literature review tables with active links, master comparison table, mathematical derivations, and quantitative benchmark results). |
| **`REPORT.md`** | Source Markdown version of the technical report with all embedded figures. |
| **`research_paper/paper.md`** | Academic research paper draft with literature review, equations, and active DOI/arXiv/IEEE links. |
| **`docs/screenshots/`** | 19 real, high-resolution screenshots capturing every screen, comparison slider, binarization mode, and diagnostic stage. |
| **`setup.bat`** | **1-Click Environment Setup** (creates virtual environment and installs all dependencies). |
| **`run_app.bat`** | **1-Click Web Studio Launcher** (starts Flask server and opens default web browser). |
| **`launch_app.bat`** | **1-Click Streamlit Studio Launcher** (starts Streamlit diagnostic UI on IPv4 `127.0.0.1`). |
| **`evaluate.bat`** | **1-Click Benchmark Evaluator** (runs quantitative evaluation across all smartphone presets). |
| **`server.py`** | Flask REST API backend serving the Web Studio and processing pipeline. |
| **`app.py`** | Streamlit analytics application with gauge meters and debug stage inspection. |
| **`pipeline.py`** | Master pipeline orchestrator executing Phases 1 through 8. |
| **`src/`** | Modular computer vision architecture:
- `phase1_input.py`: Image loading & EXIF orientation alignment
- `phase2_illumination.py`: Illumination preconditioning
- `phase3_page_seg.py`: Border-seeded GrabCut segmentation & quad detection
- `phase4_perspective.py`: Zero-leakage masking & homography warping
- `phase5_dewarping.py`: Coupled-snake baseline tracking & cylindrical uncurling
- `phase6_enhancement.py`: Multi-scale flat-field division & 3-channel scanning modes
- `phase7_quality.py`: Tenengrad sharpness, RMS contrast, and straightness metrics
- `phase8_export.py`: High-resolution scan, comparison, and PDF report generation |
| **`static/`** | Web Studio UI assets (`index.html`, `app.js`, `style.css`). |
| **`evaluate.py`** | Benchmark evaluation script computing the quantitative performance matrix. |
| **`outputs/evaluation_results/`** | Benchmark comparison artifacts, CSV performance matrix, and JSON evaluation metadata. |
| **`requirements.txt`** | Full Python dependencies list. |

---

### 3. Step-by-Step Execution Guide

#### Step 1: One-Click Environment Setup (First-Time Only)
Double-click:
```bat
setup.bat
```
This script automatically:
1. Verifies that Python (3.9+) is present on your system.
2. Creates a clean local virtual environment (`.venv`).
3. Installs all required packages from `requirements.txt` (`opencv`, `numpy`, `scipy`, `scikit-image`, `flask`, `streamlit`, `fpdf2`, `plotly`, `python-docx`, etc.).

---

#### Step 2: Launch the System

##### Option A: Launch Interactive Web Studio (Recommended)
Double-click:
```bat
run_app.bat
```
*(or `launch_webapp.bat`)*
- Automatically starts the backend server on `http://127.0.0.1:5000`.
- Opens your default web browser to the Web Studio interface.

##### Option B: Launch Streamlit Diagnostic Studio
Double-click:
```bat
launch_app.bat
```
- Starts the Streamlit application on `http://127.0.0.1:8501`.
- Provides intermediate pipeline stage inspection and radial gauge analysis.

---

### 4. Interactive Testing Instructions

#### In the Web Studio (`http://127.0.0.1:5000`):
1. **Testing Smartphone Presets**:
   - In the left sidebar, click on any preloaded sample document (`IMG_3712.jpeg` through `IMG_3718.jpeg`).
   - Click **"Scan Document"**.
   - Use the **interactive before/after slider** to drag between the original warped camera photograph and the 100% dewarped, clean scan.
   - Switch between **Clean Color Scan**, **Clean Black & White**, and **Dewarped Photo** with zero latency.
2. **Testing Custom Image Uploads ("Add Image")**:
   - Click **"Choose Image"** or drag-and-drop any custom smartphone photograph (e.g., any document on your desk or bedsheet).
   - If the photo was taken in landscape/sideways orientation, use the **Rotate Document** buttons ($90^\circ, 180^\circ, 270^\circ$).
   - Click **"Scan Document"** to execute the pipeline.
3. **Exporting Results**:
   - Click **"Download High-Res Scan"** to save the clean PNG scan.
   - Click **"Download PDF Quality Report"** to export a formal diagnostic report.

---

### 5. Running the Quantitative Evaluation Benchmark

To run the automated benchmark across all sample documents and regenerate the performance matrix:
Double-click:
```bat
evaluate.bat
```
*(or run `python evaluate.py` from the command line)*

This script prints the quantitative performance table and saves output artifacts to `outputs/evaluation_results/`:
- `comparison_IMG_3712.jpeg.jpg` through `comparison_IMG_3718.jpeg.jpg`
- `performance_matrix.csv`
- `performance_matrix.json`

---

### 6. Troubleshooting & System Requirements
- **Operating System**: Windows 10/11, macOS, or Linux.
- **Python**: Python 3.9, 3.10, 3.11, or 3.12 (Python 3.10+ recommended).
- **Tesseract OCR (Optional)**: If installed and added to PATH, Tesseract will automatically provide OCR confidence scores. If not installed, the pipeline operates normally and estimates OCR readiness via gradient and contrast analysis.
