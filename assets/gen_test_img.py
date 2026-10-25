"""
test_page_detection.py — Test benchmark for page detection on complex backgrounds
"""
import cv2
import numpy as np
from pathlib import Path

def create_complex_test_image():
    h, w = 1200, 900
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 1. Red desk surface background (like user's photo)
    canvas[:, :] = (30, 20, 150) # Dark red BGR
    
    # 2. Woven desk mat texture on left side
    for y in range(0, h, 8):
        for x in range(0, 250, 8):
            v = (x ^ y) % 60
            canvas[y:y+8, x:x+8] = (100 + v, 120 + v, 100 + v)
            
    # 3. Poster / text at top
    cv2.rectangle(canvas, (250, 20), (550, 200), (200, 180, 50), -1)
    cv2.putText(canvas, "HEADER BANNER", (270, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # 4. Red cable on right
    cv2.polylines(canvas, [np.array([[750, 0], [800, 300], [850, 700], [880, 1200]])], False, (10, 10, 200), 12)
    
    # 5. White Paper Sheet in Center (rotated/warped quad)
    paper_pts = np.array([
        [200, 280],  # TL
        [720, 260],  # TR
        [780, 1100], # BR
        [150, 1050]  # BL
    ], dtype=np.int32)
    
    # Fill white paper with off-white color
    cv2.fillConvexPoly(canvas, paper_pts, (235, 240, 245))
    
    # Add handwritten text lines on paper
    lines = [
        "VJTI - 3012 CS 510",
        "DJ - 3199",
        "Bhagubhai - 3025",
        "Thakur Polytech - 3143",
        "Tanisha - 6000"
    ]
    for i, line in enumerate(lines):
        y_pos = 350 + i * 120
        cv2.putText(canvas, line, (240, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 30, 20), 2)
        
    return canvas, paper_pts

if __name__ == "__main__":
    img, gt_pts = create_complex_test_image()
    Path("assets").mkdir(exist_ok=True)
    cv2.imwrite("assets/complex_test.png", img)
    print("Complex test image created at assets/complex_test.png")
