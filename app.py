"""
app.py — Streamlit Web UI
Real-Time Intelligent Document Dewarping & Scan Quality Assessment

Pages:
  1. 📄 Upload & Process  — upload, run pipeline, before/after
  2. 📊 Quality Report    — metric gauges and recommendations
  3. 🔬 Debug View        — intermediate pipeline stages
  4. 📈 Batch Evaluate    — run on dataset folder
  5. 📚 About             — project info & paper abstract
"""
import sys
import io

# Force stdout/stderr to UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import tempfile
from pathlib import Path
from PIL import Image

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="DocDewarp — Intelligent Document Scanner",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (Clean, Professional, Minimal) ────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }

  .metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.4rem 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }
  .metric-label { font-size: 0.75rem; color: #64748b; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }
  .metric-value { font-size: 1.5rem; font-weight: 700; color: #0f172a; margin-top: 0.2rem; }
  .metric-badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; margin-top: 0.3rem; }
  .badge-good  { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
  .badge-warn  { background: #fefce8; color: #854d0e; border: 1px solid #fef08a; }
  .badge-bad   { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }

  .hero-banner {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.2rem;
  }
  .hero-title { font-size: 1.4rem; font-weight: 700; color: #0f172a; margin: 0; }
  .hero-sub   { font-size: 0.9rem; color: #64748b; margin-top: 0.2rem; }

  .stage-chip {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 0.25rem 0.7rem;
    font-size: 0.78rem;
    font-weight: 500;
    color: #334155;
    margin: 0.15rem;
  }

  .score-label { text-align: center; font-size: 1rem; color: #64748b; }
  .star-rating { font-size: 1.8rem; text-align: center; letter-spacing: 0.1em; }

  /* Info boxes */
  .info-box {
    background: rgba(60,60,160,0.15);
    border-left: 3px solid #5050ff;
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1.1rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    color: #b0b0e0;
  }
  .rec-box {
    background: rgba(0,180,80,0.08);
    border-left: 3px solid #00c060;
    border-radius: 0 10px 10px 0;
    padding: 0.7rem 1rem;
    margin: 0.3rem 0;
    color: #80ffb0;
    font-size: 0.88rem;
  }

  /* Upload area */
  [data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04) !important;
    border: 2px dashed rgba(100,100,220,0.4) !important;
    border-radius: 14px !important;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #4834d4, #7050ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s !important;
    letter-spacing: 0.03em !important;
  }
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(80,52,212,0.5) !important;
  }

  /* Tabs */
  [data-testid="stTabs"] button {
    color: #8080b0 !important;
    font-weight: 500 !important;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color: #a0b0ff !important;
    border-bottom-color: #6050ff !important;
  }

  /* Divider */
  hr { border-color: rgba(120,120,220,0.2) !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0f0c29; }
  ::-webkit-scrollbar-thumb { background: #4040a0; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV) to RGB (Streamlit/PIL)."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def np_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr_to_rgb(img))


def load_uploaded(uploaded_file) -> np.ndarray | None:
    """Convert Streamlit uploaded file to OpenCV BGR array."""
    try:
        data = np.frombuffer(uploaded_file.read(), np.uint8)
        img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def badge(label: str, value: str, kind: str = "good") -> str:
    cls = {"good": "badge-good", "warn": "badge-warn", "bad": "badge-bad"}.get(kind, "badge-good")
    return (f'<div class="metric-card">'
            f'  <div class="metric-label">{label}</div>'
            f'  <div class="metric-value">{value}</div>'
            f'  <span class="metric-badge {cls}">{value}</span>'
            f'</div>')


def metric_card(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="metric-card">'
            f'  <div class="metric-label">{label}</div>'
            f'  <div class="metric-value">{value}</div>'
            f'  {"<div style=\"color:#7070a0;font-size:0.8rem;margin-top:0.2rem\">"+sub+"</div>" if sub else ""}'
            f'</div>')


def make_gauge(value: float, title: str, max_val: float = 100) -> go.Figure:
    """Create a Plotly gauge chart."""
    pct = value / max_val * 100
    color = "#00e070" if pct >= 70 else "#ffcc40" if pct >= 40 else "#ff6060"

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = value,
        title = {"text": title, "font": {"size": 13, "color": "#a0a0d0"}},
        number= {"suffix": f"/{max_val:.0f}", "font": {"size": 18, "color": color}},
        gauge = {
            "axis":    {"range": [0, max_val], "tickcolor": "#6060a0"},
            "bar":     {"color": color},
            "bgcolor": "rgba(30,30,70,0.5)",
            "steps":   [
                {"range": [0,    max_val*0.4], "color": "rgba(200,60,60,0.15)"},
                {"range": [max_val*0.4, max_val*0.7], "color": "rgba(255,180,0,0.15)"},
                {"range": [max_val*0.7, max_val],    "color": "rgba(0,200,100,0.15)"},
            ],
            "threshold": {"line": {"color": color, "width": 3},
                          "thickness": 0.8, "value": value},
        }
    ))
    fig.update_layout(
        height=200, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#a0a0d0"
    )
    return fig


def label_badge_kind(label: str, good_val: str, warn_val: str | None = None) -> str:
    if label == good_val:
        return "good"
    if warn_val and label == warn_val:
        return "warn"
    return "bad"


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("## 📄 DocDewarp")
        st.markdown("*Intelligent Document Scanner*")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["📄 Upload & Process",
             "📊 Quality Report",
             "🔬 Debug View",
             "📈 Batch Evaluate",
             "📚 About"],
            label_visibility="collapsed"
        )
        st.markdown("---")

        st.markdown("#### ⚙️ Settings")
        run_dewarp  = st.toggle("Enable Dewarping",   value=True, help="Coupled-snake dewarping (Phase 5)")
        run_enhance = st.toggle("Enable Enhancement", value=True, help="Sharpening + shadow removal (Phase 6)")

        st.markdown("#### 🌐 OCR Language")
        ocr_lang = st.selectbox("Language", ["eng", "eng+hin", "eng+ara", "hin", "deu", "fra"],
                                 help="Tesseract language for OCR accuracy measurement")

        st.markdown("---")
        st.markdown(
            '<div style="color:#6060a0;font-size:0.75rem;text-align:center">'
            'Classical CV · No Deep Learning<br>'
            'Based on Bukhari et al. (2009)'
            '</div>',
            unsafe_allow_html=True
        )

    return page, run_dewarp, run_enhance, ocr_lang


# ══════════════════════════════════════════════════════════════════════════════
# Page 1 — Upload & Process
# ══════════════════════════════════════════════════════════════════════════════

def page_upload(run_dewarp, run_enhance, ocr_lang):
    st.markdown("""
    <div class="hero-banner">
      <div class="hero-title">📄 Document Dewarping Pipeline</div>
      <div class="hero-sub">Upload a warped or skewed document to correct, enhance, and assess quality automatically.</div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline stage chips
    stages = ["①  Preprocess", "②  Detect Page", "③  Perspective",
              "④  Dewarp" if run_dewarp else "④  Skip Dewarp",
              "⑤  Enhance" if run_enhance else "⑤  Skip Enhance",
              "⑥  Quality Score", "⑦  Export"]
    chips = "".join(f'<span class="stage-chip">{s}</span>' for s in stages)
    st.markdown(chips, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop your document image here",
        type=["jpg", "jpeg", "png", "tiff", "bmp"],
        help="Supports JPG, PNG, TIFF, BMP"
    )

    if uploaded is None:
        st.markdown("""
        <div class="info-box">
          ℹ️  Upload a scanned or photographed document above to begin.
          The pipeline will detect the page, correct perspective, dewarp curves, and generate a quality report.
        </div>
        """, unsafe_allow_html=True)
        return

    img = load_uploaded(uploaded)
        
    if img is None:
        st.error("❌ Could not read the file. Please try another image.")
        return

    st.session_state["original_img"] = img

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run_btn = st.button("Scan & Dewarp Document", use_container_width=True)

    if not run_btn:
        st.markdown("#### Preview — Original Image")
        st.image(np_to_pil(img), use_container_width=True)
        return

    # ── Run pipeline ──────────────────────────────────────────────────────────
    with st.spinner("Processing document..."):
        progress = st.progress(0, text="Starting pipeline...")
        status   = st.empty()
        
        from pipeline import run_pipeline
        
        progress.progress(10, "Phase 2: Preprocessing...")
        status.info("🔄 Running pipeline — this may take a few seconds...")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, img)

    start = time.perf_counter()
    result = run_pipeline(
        tmp_path,
        output_dir   = "outputs",
        save_outputs = True,
        run_dewarp   = run_dewarp,
        run_enhance  = run_enhance,
        verbose      = False,
    )
    elapsed = time.perf_counter() - start

    progress.progress(100, "✅ Pipeline complete!")
    status.success(f"✅ Pipeline completed in **{elapsed:.2f}s**")

    st.session_state["pipeline_result"] = result

    if not result.success:
        st.error(f"Pipeline error: {result.error_message}")
        return

    # ── Before / After ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Before / After")

    if result.enhanced and result.enhanced.get("natural_scan") is not None:
        final_img = result.enhanced["natural_scan"]
    elif result.enhanced and result.enhanced.get("magic_color") is not None:
        final_img = result.enhanced["magic_color"]
    elif result.enhanced and result.enhanced.get("sharpened_color") is not None:
        final_img = result.enhanced["sharpened_color"]
    elif result.dewarped is not None:
        final_img = result.dewarped
    elif result.perspective_corrected is not None:
        final_img = result.perspective_corrected
    else:
        final_img = result.original

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🔴 Original (Warped)**")
        st.image(np_to_pil(img), use_container_width=True)
    with c2:
        st.markdown("**🟢 Corrected (Dewarped)**")
        st.image(np_to_pil(final_img), use_container_width=True)

    # ── Quick metrics ─────────────────────────────────────────────────────────
    if result.quality_after:
        qa = result.quality_after
        st.markdown("---")
        st.markdown("### Quick Quality Summary")
        cols = st.columns(4)
        with cols[0]: st.metric("Overall Score", f"{qa.overall_score:.0f}/100")
        with cols[1]: st.metric("Sharpness",     f"{qa.sharpness_score:.1f}%")
        with cols[2]: st.metric("OCR Readiness", f"{qa.ocr_readiness:.1f}%")
        with cols[3]: st.metric("Stars",         "★" * qa.star_rating + "☆" * (5 - qa.star_rating))

    st.markdown("""
    > 💡 **Tip:** Switch to the **📊 Quality Report** tab for detailed metrics and gauges,
    > or **🔬 Debug View** to inspect intermediate pipeline stages.
    """)

    # ── Download buttons ──────────────────────────────────────────────────────
    if result.saved_image_path and Path(result.saved_image_path).exists():
        st.markdown("---")
        st.markdown("### 💾 Download Outputs")
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            with open(result.saved_image_path, "rb") as f:
                st.download_button("⬇️ Download Corrected Image", f,
                                   file_name="corrected_document.png",
                                   mime="image/png")
        if result.saved_pdf_path and Path(result.saved_pdf_path).exists():
            with dl_col2:
                with open(result.saved_pdf_path, "rb") as f:
                    st.download_button("⬇️ Download Quality Report (PDF)", f,
                                       file_name="quality_report.pdf",
                                       mime="application/pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Page 2 — Quality Report
# ══════════════════════════════════════════════════════════════════════════════

def page_quality():
    st.markdown("## 📊 Document Quality Report")

    result = st.session_state.get("pipeline_result")
    if not result or not result.quality_after:
        st.markdown("""
        <div class="info-box">
          ℹ️  Process a document first on the <strong>Upload & Process</strong> page.
        </div>
        """, unsafe_allow_html=True)
        return

    qa = result.quality_after
    qb = result.quality_before

    # ── Score ring ────────────────────────────────────────────────────────────
    col_score, col_info = st.columns([1, 2])
    with col_score:
        fig = make_gauge(qa.overall_score, "Overall Quality", 100)
        st.plotly_chart(fig, use_container_width=True)

        stars = "★" * qa.star_rating + "☆" * (5 - qa.star_rating)
        st.markdown(f'<div class="star-rating" style="color:#ffcc40">{stars}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="score-label">{qa.star_rating}/5 Stars</div>',
                    unsafe_allow_html=True)

    with col_info:
        if qb:
            delta = qa.overall_score - qb.overall_score
            arrow = "↑" if delta >= 0 else "↓"
            color = "#00e070" if delta >= 0 else "#ff6060"
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Score Improvement</div>'
                f'<div class="metric-value" style="color:{color}">{arrow} {abs(delta):.1f} pts</div>'
                f'<div style="color:#7070a0;font-size:0.8rem">{qb.overall_score:.0f} → {qa.overall_score:.0f}</div></div>',
                unsafe_allow_html=True
            )
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">OCR Readiness</div>'
            f'<div class="metric-value">{qa.ocr_readiness:.1f}%</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Gauges row ────────────────────────────────────────────────────────────
    st.markdown("### 📈 Metric Gauges")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(make_gauge(qa.sharpness_score, "Sharpness"), use_container_width=True)
    with g2:
        st.plotly_chart(make_gauge(qa.ocr_readiness, "OCR Readiness"), use_container_width=True)
    with g3:
        contrast_score = {"Excellent": 90, "Good": 65, "Low": 30}.get(qa.contrast_ok, 50)
        st.plotly_chart(make_gauge(contrast_score, "Contrast"), use_container_width=True)

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.markdown("### 🔢 All Metrics")
    c1, c2, c3 = st.columns(3)

    with c1:
        blur_kind = label_badge_kind(qa.blur_level,    "Low", "Medium")
        st.markdown(badge("Blur Level", qa.blur_level, blur_kind), unsafe_allow_html=True)

        bright_kind = label_badge_kind(qa.brightness_ok, "Good")
        st.markdown(badge("Brightness", qa.brightness_ok, bright_kind), unsafe_allow_html=True)

    with c2:
        cont_kind = label_badge_kind(qa.contrast_ok, "Excellent", "Good")
        st.markdown(badge("Contrast", qa.contrast_ok, cont_kind), unsafe_allow_html=True)

        noise_kind = label_badge_kind(qa.noise_level, "Low", "Medium")
        st.markdown(badge("Noise Level", qa.noise_level, noise_kind), unsafe_allow_html=True)

    with c3:
        persp_kind = "good" if qa.perspective_ok else "bad"
        st.markdown(badge("Perspective", "Correct" if qa.perspective_ok else "Distorted", persp_kind),
                    unsafe_allow_html=True)
        st.markdown(
            metric_card("Laplacian Variance", f"{qa.blur_variance:.1f}",
                        "Higher = sharper"),
            unsafe_allow_html=True
        )

    # ── Radar chart ───────────────────────────────────────────────────────────
    if qb:
        st.markdown("---")
        st.markdown("### ⚡ Before vs After Radar")
        cats = ["Sharpness", "Blur Score", "Brightness", "Contrast", "OCR Readiness"]

        blur_score = lambda b: {"Low": 90, "Medium": 55, "High": 15}.get(b, 50)
        bright_score = lambda b: {"Good": 90, "Too Dark": 30, "Too Bright": 30}.get(b, 50)
        cont_score  = lambda c: {"Excellent": 90, "Good": 65, "Low": 25}.get(c, 50)

        before_vals = [qb.sharpness_score, blur_score(qb.blur_level),
                       bright_score(qb.brightness_ok), cont_score(qb.contrast_ok),
                       qb.ocr_readiness]
        after_vals  = [qa.sharpness_score, blur_score(qa.blur_level),
                       bright_score(qa.brightness_ok), cont_score(qa.contrast_ok),
                       qa.ocr_readiness]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=before_vals + [before_vals[0]], theta=cats + [cats[0]],
                                       fill="toself", name="Before",
                                       line_color="#DD8452", fillcolor="rgba(220,130,80,0.2)"))
        fig.add_trace(go.Scatterpolar(r=after_vals + [after_vals[0]], theta=cats + [cats[0]],
                                       fill="toself", name="After",
                                       line_color="#4C72B0", fillcolor="rgba(76,114,176,0.2)"))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100],
                                        gridcolor="rgba(120,120,200,0.2)",
                                        tickfont=dict(color="#8080b0"))),
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a0a0d0",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Recommendations ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💡 Auto-Fix Recommendations")
    for rec in qa.recommendations:
        st.markdown(f'<div class="rec-box">✅ {rec}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Page 3 — Debug View
# ══════════════════════════════════════════════════════════════════════════════

def page_debug():
    st.markdown("## 🔬 Pipeline Debug View")
    st.markdown("*Inspect intermediate outputs at each stage.*")

    result = st.session_state.get("pipeline_result")
    if not result:
        st.markdown("""
        <div class="info-box">ℹ️  Process a document first on the Upload & Process page.</div>
        """, unsafe_allow_html=True)
        return

    tabs = st.tabs(["Preprocessing", "Page Detection", "Perspective",
                     "Dewarping", "Enhancement", "Timings"])

    # ── Preprocessing ──────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("#### Phase 2 — Preprocessing Stages")
        prep = result.preprocessed
        if prep:
            stage_names = ["gray", "bilateral", "clahe", "denoised"]
            cols = st.columns(len(stage_names))
            for col, name in zip(cols, stage_names):
                if name in prep:
                    with col:
                        st.markdown(f"**{name.upper()}**")
                        st.image(prep[name], use_container_width=True, clamp=True)

    # ── Page Detection ─────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("#### Phase 3 — Page Quad Detection")
        if result.quad_overlay is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Detected Quadrilateral**")
                st.image(np_to_pil(result.quad_overlay), use_container_width=True)
            with col2:
                st.markdown("**Detection Status**")
                if result.page_detected:
                    st.success("✅ Page quadrilateral detected successfully")
                    if result.quad is not None:
                        corners = result.quad.reshape(4, 2)
                        df = pd.DataFrame(corners, columns=["X", "Y"],
                                          index=["TL","TR","BR","BL"])
                        st.dataframe(df, use_container_width=True)
                else:
                    st.warning("⚠️ No clear page quad found — full image used")

    # ── Perspective ────────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("#### Phase 4 — Perspective Correction")
        if result.perspective_corrected is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**After Perspective Correction**")
                st.image(np_to_pil(result.perspective_corrected), use_container_width=True)
            with c2:
                if result.original is not None:
                    h_orig, w_orig = result.original.shape[:2]
                    h_corr, w_corr = result.perspective_corrected.shape[:2]
                    st.markdown("**Dimensions**")
                    st.markdown(f"- Original: `{w_orig} × {h_orig}`")
                    st.markdown(f"- Corrected: `{w_corr} × {h_corr}`")

    # ── Dewarping ──────────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("#### Phase 5 — Coupled-Snake Dewarping")
        dbg = result.dewarp_debug
        if dbg:
            n_lines = dbg.get("n_lines", 0)
            st.metric("Text Lines Detected", n_lines)

            if "snake_overlay" in dbg:
                st.markdown("**Snake Overlay (Upper=Blue, Lower=Orange, Mid=Green)**")
                st.image(np_to_pil(dbg["snake_overlay"]), use_container_width=True)
            elif result.dewarped is not None:
                st.markdown("**Dewarped Output**")
                st.image(np_to_pil(result.dewarped), use_container_width=True)

            if "binary" in dbg:
                st.markdown("**Binarized Image (for text-line detection)**")
                st.image(dbg["binary"], use_container_width=True, clamp=True)
        else:
            if result.dewarped is not None:
                st.image(np_to_pil(result.dewarped), use_container_width=True)

    # ── Enhancement ────────────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("#### Phase 6 — Image Enhancement")
        enh = result.enhanced
        if enh:
            selected = st.selectbox("Select enhancement output",
                                     [k for k in enh.keys()])
            img_show = enh[selected]
            if isinstance(img_show, np.ndarray):
                if len(img_show.shape) == 3:
                    st.image(np_to_pil(img_show), use_container_width=True)
                else:
                    st.image(img_show, use_container_width=True, clamp=True)

    # ── Timings ────────────────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown("#### ⏱ Phase Timings")
        if result.timings:
            timing_data = {k: v*1000 for k, v in result.timings.items() if v > 0}
            df = pd.DataFrame(list(timing_data.items()), columns=["Phase", "Time (ms)"])
            df = df.sort_values("Time (ms)", ascending=True)

            fig = px.bar(df, x="Time (ms)", y="Phase", orientation="h",
                         color="Time (ms)", color_continuous_scale="Blues",
                         title=f"Total: {result.total_time*1000:.0f} ms")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(20,20,60,0.3)",
                               font_color="#a0a0d0", height=300)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df.sort_values("Time (ms)", ascending=False),
                         use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Page 4 — Batch Evaluate
# ══════════════════════════════════════════════════════════════════════════════

def page_batch():
    st.markdown("## 📈 Batch Evaluation")
    st.markdown("Run the full evaluation suite on the dataset.")

    manifest_path = Path("dataset/dataset_manifest.json")
    csv_path      = Path("experiments/results.csv")

    col1, col2 = st.columns(2)
    with col1:
        if manifest_path.exists():
            import json
            with open(manifest_path) as f:
                manifest = json.load(f)
            st.success(f"✅ Dataset manifest found — {manifest['total']} image pairs")
        else:
            st.warning("⚠️ No manifest found. Generate a dataset first.")
            st.code("python dataset_generator.py --synthetic", language="bash")

    with col2:
        max_imgs = st.slider("Max images to evaluate", 5, 100, 20)
        ocr_lang = st.selectbox("OCR Language", ["eng", "eng+hin", "deu", "fra"])

    if st.button("▶️ Run Evaluation", use_container_width=False):
        if not manifest_path.exists():
            st.error("Generate the dataset first!")
            return

        with st.spinner("Running evaluation — this may take several minutes..."):
            from evaluate import run_evaluation
            df = run_evaluation(
                manifest_path = manifest_path,
                output_dir    = "experiments",
                ocr_lang      = ocr_lang,
                max_images    = max_imgs,
            )

        if not df.empty:
            st.success(f"✅ Evaluated {len(df)} images. Results saved to `experiments/results.csv`")
            st.session_state["eval_df"] = df

    # Show results if available
    df = st.session_state.get("eval_df")
    if df is None and csv_path.exists():
        df = pd.read_csv(csv_path)
        st.session_state["eval_df"] = df

    if df is not None and not df.empty:
        st.markdown("---")
        st.markdown("### 📊 Evaluation Results")

        # Summary table
        summary = pd.DataFrame({
            "Metric": ["SSIM", "PSNR (dB)", "Straightness", "Quality Score"],
            "Full Pipeline": [
                round(float(df["full_ssim"].mean()), 4),
                round(float(df["full_psnr"].mean()), 2),
                round(float(df["full_straightness"].mean()), 2),
                round(float(df["full_quality"].mean()), 2),
            ],
            "Baseline": [
                round(float(df["base_ssim"].mean()), 4),
                round(float(df["base_psnr"].mean()), 2),
                round(float(df["base_straightness"].mean()), 2),
                round(float(df["base_quality"].mean()), 2),
            ],
            "Improvement": [
                round(float(df["delta_ssim"].mean()), 4),
                round(float(df["delta_psnr"].mean()), 2),
                round(float((df["full_straightness"] - df["base_straightness"]).mean()), 2),
                round(float(df["delta_quality"].mean()), 2),
            ]
        })
        st.dataframe(summary, use_container_width=True)

        # Charts
        chart_path = Path("experiments/charts/evaluation_charts.png")
        if chart_path.exists():
            st.image(str(chart_path), use_container_width=True,
                     caption="Evaluation Charts")

        # Raw data
        with st.expander("📋 Raw Results Table"):
            st.dataframe(df, use_container_width=True)

        if csv_path.exists():
            with open(csv_path, "rb") as f:
                st.download_button("⬇️ Download results.csv", f,
                                   file_name="evaluation_results.csv",
                                   mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# Page 5 — About
# ══════════════════════════════════════════════════════════════════════════════

def page_about():
    st.markdown("## 📚 About This Project")

    st.markdown("""
    <div class="metric-card">
      <div class="metric-label">Project Title</div>
      <div class="metric-value" style="font-size:1.2rem">Real-Time Intelligent Document Dewarping & Scan Quality Assessment</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📖 Research Foundation")
    st.markdown("""
    This project implements and extends the approach from:

    > **Bukhari, S.S., Shafait, F., & Breuel, T.M. (2009)**
    > *Dewarping of Document Images using Coupled-Snakes*
    > ResearchGate — Document Analysis & Recognition

    The core idea: model page curvature using **coupled active contours (snakes)**
    fitted to the upper and lower boundaries of detected text lines. The resulting
    displacement map is used to "unfold" the curved page.
    """)

    st.markdown("---")
    st.markdown("### 🏗️ Pipeline Architecture")

    pipeline_steps = [
        ("Phase 1", "Input", "Load image from file upload"),
        ("Phase 2", "Preprocessing", "Grayscale → Bilateral → CLAHE → Gaussian"),
        ("Phase 3", "Page Detection", "Canny → Morphology → Contour → Quad"),
        ("Phase 4", "Perspective", "4-point homography correction"),
        ("Phase 5", "Dewarping", "Text-line detection + Coupled Snakes + Remap"),
        ("Phase 6", "Enhancement", "Unsharp mask + Shadow removal + Deskew"),
        ("Phase 7", "Quality Analysis", "Sharpness, Blur, SSIM, OCR readiness"),
        ("Phase 8", "Export", "PNG + PDF quality report"),
    ]
    df = pd.DataFrame(pipeline_steps, columns=["Phase", "Name", "Method"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📐 Evaluation Metrics")
    metrics_info = [
        ("SSIM", "Structural Similarity Index vs ground truth"),
        ("PSNR", "Peak Signal-to-Noise Ratio vs ground truth"),
        ("Text Straightness", "Std dev of Hough line angles (lower = straighter)"),
        ("OCR Confidence", "Tesseract character confidence (0–100)"),
        ("Sharpness", "Variance of Laplacian operator"),
        ("Runtime", "Per-phase timing via perf_counter()"),
    ]
    for name, desc in metrics_info:
        st.markdown(f'<div class="info-box"><strong>{name}</strong> — {desc}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    | Component | Technology |
    |---|---|
    | Language | Python 3.10+ |
    | Core CV | OpenCV 4.x |
    | Numerics | NumPy, SciPy |
    | Quality | scikit-image |
    | OCR | pytesseract + Tesseract |
    | UI | Streamlit + Plotly |
    | Reports | FPDF2 |
    | No deep learning | ✅ Pure classical CV |
    """)


# ══════════════════════════════════════════════════════════════════════════════
# Main App
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Initialize session state
    if "pipeline_result" not in st.session_state:
        st.session_state["pipeline_result"] = None
    if "original_img" not in st.session_state:
        st.session_state["original_img"] = None
    if "eval_df" not in st.session_state:
        st.session_state["eval_df"] = None

    page, run_dewarp, run_enhance, ocr_lang = render_sidebar()

    if page == "📄 Upload & Process":
        page_upload(run_dewarp, run_enhance, ocr_lang)
    elif page == "📊 Quality Report":
        page_quality()
    elif page == "🔬 Debug View":
        page_debug()
    elif page == "📈 Batch Evaluate":
        page_batch()
    elif page == "📚 About":
        page_about()


if __name__ == "__main__":
    main()
