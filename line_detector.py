import cv2
import numpy as np

def detectar_linea_continua(frame, min_length=200):
    # Convert frame to grayscale for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect edges using Canny
    edges = cv2.Canny(gray, 80, 150)

    # Detect line segments using Hough transform
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi/180,
        threshold=60,
        minLineLength=min_length,
        maxLineGap=20
    )

    if lines is None:
        # No line candidates detected
        return None

    # Select the longest detected line
    best = None
    best_len = 0
    for x1, y1, x2, y2 in lines[:,0]:
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if length > best_len:
            best_len = length
            best = (x1, y1, x2, y2)

    return best
