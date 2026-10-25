/**
 * app.js — Interactive Document Scanner & Dewarping Web Studio
 * Clean, lightweight, zero-dependency vanilla JS client.
 */

document.addEventListener("DOMContentLoaded", () => {
  // State
  const state = {
    selectedSample: "IMG_3712.jpeg",
    uploadedFile: null,
    rotation: 0,
    scanMode: "clean_color",
    trimMargin: 0.015,
    dewarp: true,
    activeView: "split",
    processedData: null,
    isProcessing: false,
  };

  // DOM Elements
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const samplesList = document.getElementById("samplesList");
  const modeSelector = document.getElementById("modeSelector");
  const modeHint = document.getElementById("modeHint");
  const btnRotateCw = document.getElementById("btnRotateCw");
  const btnRotateCcw = document.getElementById("btnRotateCcw");
  const btnResetRot = document.getElementById("btnResetRot");
  const rotIndicator = document.getElementById("rotIndicator");
  const chkDewarp = document.getElementById("chkDewarp");
  const chkTrimMargin = document.getElementById("chkTrimMargin");
  const btnProcess = document.getElementById("btnProcess");
  const viewTabs = document.getElementById("viewTabs");
  const statusBadge = document.getElementById("statusBadge");

  // View containers
  const emptyState = document.getElementById("emptyState");
  const spinnerOverlay = document.getElementById("spinnerOverlay");
  const spinnerText = document.getElementById("spinnerText");
  const sliderContainer = document.getElementById("sliderContainer");
  const sliderAfterWrap = document.getElementById("sliderAfterWrap");
  const sliderHandle = document.getElementById("sliderHandle");
  const imgBefore = document.getElementById("imgBefore");
  const imgAfter = document.getElementById("imgAfter");
  const singleViewContainer = document.getElementById("singleViewContainer");
  const imgSingle = document.getElementById("imgSingle");
  const sideBySideContainer = document.getElementById("sideBySideContainer");
  const imgSideOriginal = document.getElementById("imgSideOriginal");
  const imgSideScanned = document.getElementById("imgSideScanned");

  // Actions & Metrics
  const btnDownloadPng = document.getElementById("btnDownloadPng");
  const btnDownloadPdf = document.getElementById("btnDownloadPdf");
  const metricsCard = document.getElementById("metricsCard");
  const valQualityAfter = document.getElementById("valQualityAfter");
  const valQualityDelta = document.getElementById("valQualityDelta");
  const valQualityBefore = document.getElementById("valQualityBefore");
  const valSharpness = document.getElementById("valSharpness");
  const valBlurLevel = document.getElementById("valBlurLevel");
  const valContrast = document.getElementById("valContrast");
  const valOcr = document.getElementById("valOcr");
  const valTime = document.getElementById("valTime");
  const valTimingBreakdown = document.getElementById("valTimingBreakdown");

  // ───────────────────────────────────────────────────────────────────────────
  // 1. Initialize & Fetch Preset Samples
  // ───────────────────────────────────────────────────────────────────────────
  async function loadSamples() {
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      if (!data.samples || data.samples.length === 0) {
        samplesList.innerHTML = `<div class="sample-desc">No presets found.</div>`;
        return;
      }
      samplesList.innerHTML = "";
      data.samples.forEach((s) => {
        const item = document.createElement("div");
        item.className = `sample-item ${s.name === state.selectedSample ? "active" : ""}`;
        item.dataset.name = s.name;
        item.innerHTML = `
          <img class="sample-thumb" src="${s.path}" alt="${s.name}" loading="lazy">
          <div class="sample-info">
            <span class="sample-name">${s.title}</span>
            <span class="sample-desc">${s.description}</span>
          </div>
        `;
        item.addEventListener("click", () => selectSample(s.name));
        samplesList.appendChild(item);
      });
    } catch (err) {
      console.error("Failed to load samples:", err);
      samplesList.innerHTML = `<div class="sample-desc">Failed to load presets.</div>`;
    }
  }

  function selectSample(name) {
    state.selectedSample = name;
    state.uploadedFile = null;
    state.rotation = 0;
    updateRotationUI();
    document.querySelectorAll(".sample-item").forEach((el) => {
      el.classList.toggle("active", el.dataset.name === name);
    });
    statusBadge.textContent = `Selected: ${name}`;
    statusBadge.className = "badge badge-neutral";
  }

  // ───────────────────────────────────────────────────────────────────────────
  // 2. Add Custom Image (File Upload & Drag/Drop)
  // ───────────────────────────────────────────────────────────────────────────
  dropZone.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleUploadedFile(e.target.files[0]);
    }
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleUploadedFile(e.dataTransfer.files[0]);
    }
  });

  function handleUploadedFile(file) {
    state.uploadedFile = file;
    state.selectedSample = null;
    state.rotation = 0;
    updateRotationUI();
    document.querySelectorAll(".sample-item").forEach((el) => el.classList.remove("active"));
    statusBadge.textContent = `Ready: ${file.name}`;
    statusBadge.className = "badge badge-neutral";
    dropZone.querySelector(".upload-prompt").innerHTML = `<strong>${file.name}</strong> selected`;
  }

  // ───────────────────────────────────────────────────────────────────────────
  // 3. Scan Mode & Settings Handlers
  // ───────────────────────────────────────────────────────────────────────────
  modeSelector.addEventListener("click", (e) => {
    const btn = e.target.closest(".segment-btn");
    if (!btn) return;
    document.querySelectorAll(".segment-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.scanMode = btn.dataset.mode;

    const hints = {
      clean_color: "CamScanner-quality: pure white paper background, crisp colored inks.",
      clean_bw: "High-contrast monochrome scan: pristine binarization for OCR & printing.",
      dewarped_photo: "Perspective & curve-corrected document with natural color tones.",
    };
    modeHint.textContent = hints[state.scanMode] || "";

    // Instant zero-latency mode preview if already processed
    if (state.processedData && state.processedData.images && state.processedData.images.modes) {
      const modeImg = state.processedData.images.modes[state.scanMode];
      if (modeImg) {
        state.processedData.images.scanned = modeImg;
        imgAfter.src = modeImg;
        imgSideScanned.src = modeImg;
        if (state.activeView === "scanned") {
          imgSingle.src = modeImg;
        }
      }
    }
  });

  // Rotation Controls
  btnRotateCw.addEventListener("click", () => {
    state.rotation = (state.rotation + 90) % 360;
    updateRotationUI();
  });

  btnRotateCcw.addEventListener("click", () => {
    state.rotation = (state.rotation - 90 + 360) % 360;
    updateRotationUI();
  });

  btnResetRot.addEventListener("click", () => {
    state.rotation = 0;
    updateRotationUI();
  });

  function updateRotationUI() {
    rotIndicator.textContent = `Rotation: ${state.rotation}°`;
  }

  chkDewarp.addEventListener("change", (e) => {
    state.dewarp = e.target.checked;
  });

  chkTrimMargin.addEventListener("change", (e) => {
    state.trimMargin = e.target.checked ? 0.015 : 0.0;
  });

  // ───────────────────────────────────────────────────────────────────────────
  // 4. Document Processing (Trigger Pipeline)
  // ───────────────────────────────────────────────────────────────────────────
  btnProcess.addEventListener("click", async () => {
    if (state.isProcessing) return;
    if (!state.uploadedFile && !state.selectedSample) {
      alert("Please select a sample document or add an image.");
      return;
    }

    state.isProcessing = true;
    btnProcess.disabled = true;
    spinnerOverlay.style.display = "flex";
    spinnerText.textContent = "Segmenting document boundary & dewarping...";
    statusBadge.textContent = "Processing...";
    statusBadge.className = "badge badge-neutral";

    const formData = new FormData();
    if (state.uploadedFile) {
      formData.append("file", state.uploadedFile);
    } else {
      formData.append("sample_name", state.selectedSample);
    }
    formData.append("rotation", state.rotation);
    formData.append("scan_mode", state.scanMode);
    formData.append("trim_margin", state.trimMargin);
    formData.append("dewarp", state.dewarp ? "true" : "false");

    try {
      const res = await fetch("/api/process", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!data.success) {
        throw new Error(data.error || "Failed to process document.");
      }

      state.processedData = data;
      renderResult(data);
      statusBadge.textContent = `Scanned in ${data.total_time_sec}s`;
      statusBadge.className = "badge badge-success";
    } catch (err) {
      console.error("Processing error:", err);
      alert(`Error processing document: ${err.message}`);
      statusBadge.textContent = "Error";
      statusBadge.className = "badge badge-neutral";
    } finally {
      state.isProcessing = false;
      btnProcess.disabled = false;
      spinnerOverlay.style.display = "none";
    }
  });

  // ───────────────────────────────────────────────────────────────────────────
  // 5. Render Processing Results
  // ───────────────────────────────────────────────────────────────────────────
  function renderResult(data) {
    emptyState.style.display = "none";
    metricsCard.style.display = "flex";
    btnDownloadPng.style.display = "inline-flex";
    btnDownloadPdf.style.display = "inline-flex";

    // Set images
    const orig = data.images.original;
    const scanned = data.images.scanned;

    imgBefore.src = orig;
    imgAfter.src = scanned;
    imgSideOriginal.src = orig;
    imgSideScanned.src = scanned;

    // Quality Metrics
    const qB = data.quality_before || {};
    const qA = data.quality_after || {};
    const delta = (qA.overall_score || 0) - (qB.overall_score || 0);

    valQualityAfter.textContent = `${qA.overall_score || 0}/100`;
    valQualityDelta.textContent = delta >= 0 ? `+${delta.toFixed(1)}` : `${delta.toFixed(1)}`;
    valQualityDelta.style.color = delta >= 0 ? "var(--success)" : "#dc2626";
    valQualityBefore.textContent = `Before: ${qB.overall_score || 0}/100`;

    valSharpness.textContent = `${qA.sharpness || 0}%`;
    valBlurLevel.textContent = `Blur Level: ${qA.blur_level || "Low"}`;
    valContrast.textContent = `${qA.contrast || 0} RMS`;
    valOcr.textContent = `${qA.ocr_readiness || 0}%`;

    valTime.textContent = `${data.total_time_sec}s`;
    const timings = data.timings_ms || {};
    valTimingBreakdown.textContent = `Det: ${timings.page_detection || 0}ms • Dewarp: ${timings.dewarping || 0}ms • Scan: ${timings.enhancement || 0}ms`;

    // Reset slider to center
    setSliderPosition(50);
    switchView(state.activeView);
  }

  // ───────────────────────────────────────────────────────────────────────────
  // 6. View Switching
  // ───────────────────────────────────────────────────────────────────────────
  viewTabs.addEventListener("click", (e) => {
    const btn = e.target.closest(".tab-btn");
    if (!btn) return;
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.activeView = btn.dataset.view;
    if (state.processedData) {
      switchView(state.activeView);
    }
  });

  function switchView(view) {
    if (!state.processedData) return;

    sliderContainer.style.display = "none";
    singleViewContainer.style.display = "none";
    sideBySideContainer.style.display = "none";

    const imgs = state.processedData.images;

    if (view === "split") {
      sliderContainer.style.display = "block";
      setSliderPosition(50);
    } else if (view === "scanned") {
      singleViewContainer.style.display = "flex";
      imgSingle.src = imgs.scanned;
    } else if (view === "sidebyside") {
      sideBySideContainer.style.display = "flex";
    } else if (view === "overlay") {
      singleViewContainer.style.display = "flex";
      imgSingle.src = imgs.quad_overlay || imgs.original;
    } else if (view === "original") {
      singleViewContainer.style.display = "flex";
      imgSingle.src = imgs.original;
    }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // 7. Interactive Before/After Split Slider
  // ───────────────────────────────────────────────────────────────────────────
  let isDraggingSlider = false;

  sliderHandle.addEventListener("mousedown", () => { isDraggingSlider = true; });
  window.addEventListener("mouseup", () => { isDraggingSlider = false; });
  sliderContainer.addEventListener("mousemove", onSliderDrag);

  // Touch support for mobile/trackpad
  sliderHandle.addEventListener("touchstart", () => { isDraggingSlider = true; }, { passive: true });
  window.addEventListener("touchend", () => { isDraggingSlider = false; });
  sliderContainer.addEventListener("touchmove", (e) => {
    if (!isDraggingSlider || !e.touches[0]) return;
    const rect = sliderContainer.getBoundingClientRect();
    const x = e.touches[0].clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(pct);
  }, { passive: true });

  function onSliderDrag(e) {
    if (!isDraggingSlider) return;
    const rect = sliderContainer.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(pct);
  }

  function setSliderPosition(pct) {
    sliderAfterWrap.style.width = `${pct}%`;
    sliderHandle.style.left = `${pct}%`;
  }

  // Initial load
  loadSamples();
});
