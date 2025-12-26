import cv2
import numpy as np

# Background subtractor based on adaptive Gaussian Mixture Models
fgbg = cv2.createBackgroundSubtractorMOG2()

def detectar_coche(frame):
    # Apply background subtraction to isolate moving objects
    mask = fgbg.apply(frame)
    # Reduce salt-and-pepper noise in the foreground mask
    mask = cv2.medianBlur(mask, 5)

    # Extract external contours from the foreground mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        
        return None, None

    # Select the largest contour assuming it corresponds to the vehicle
    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)

    # Reject small detections to filter noise and irrelevant objects
    if w*h < 5000:
        return None, None

    # Compute bounding box center for tracking purposes
    cx = x + w//2
    cy = y + h//2

    return (x, y, w, h), (cx, cy)
