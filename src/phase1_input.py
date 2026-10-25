"""
Phase 1 — Input Handler
Supports file path, numpy array, and webcam capture.
"""
import cv2
import numpy as np
from pathlib import Path


from PIL import Image, ImageOps


def load_image(source, rotation: int = 0) -> np.ndarray:
    """
    Load an image from various sources with automatic EXIF orientation
    and optional rotation (0, 90, 180, 270 degrees clockwise).
    
    Args:
        source: str (file path), np.ndarray, or int (camera index)
        rotation: Clockwise rotation in degrees (0, 90, 180, 270)
    
    Returns:
        BGR numpy array
    """
    if isinstance(source, np.ndarray):
        img = source.copy()
    elif isinstance(source, (str, Path)):
        path = str(source)
        try:
            pil_img = Image.open(path)
            pil_img = ImageOps.exif_transpose(pil_img)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {path}")
    elif isinstance(source, int):
        img = capture_from_webcam(camera_index=source)
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    if rotation == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif rotation == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return img


def capture_from_webcam(camera_index: int = 0) -> np.ndarray:
    """Capture a single frame from the webcam."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise RuntimeError("Failed to capture frame from webcam")
    
    return frame


def live_capture_with_detection(camera_index: int = 0):
    """
    Live webcam feed with real-time document edge overlay.
    Press SPACE to capture, ESC to quit.
    
    Returns:
        Captured frame or None
    """
    from src.phase3_page_detection import detect_page_quad
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")
    
    captured = None
    print("[INPUT] Live capture started. SPACE=capture, ESC=quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display = frame.copy()
        
        # Try to detect document quad and overlay it
        try:
            quad, _ = detect_page_quad(frame, draw=False)
            if quad is not None:
                cv2.polylines(display, [quad], True, (0, 255, 100), 3)
                cv2.putText(display, "Document detected — SPACE to capture",
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
            else:
                cv2.putText(display, "Searching for document...",
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
        except Exception:
            pass
        
        cv2.imshow("Document Scanner — Live", display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            captured = frame.copy()
            print("[INPUT] Frame captured!")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return captured


def validate_image(img: np.ndarray) -> dict:
    """Basic image sanity checks."""
    h, w = img.shape[:2]
    return {
        "height": h,
        "width": w,
        "channels": img.shape[2] if len(img.shape) == 3 else 1,
        "dtype": str(img.dtype),
        "valid": h > 100 and w > 100,
        "megapixels": round((h * w) / 1_000_000, 2)
    }
