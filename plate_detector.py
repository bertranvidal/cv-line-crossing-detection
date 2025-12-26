import cv2
import numpy as np
import pytesseract
import re

# Path to local Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def validar_string(input_string: str) -> bool:
    # Validate Spanish license plate format: 4 digits + 3 uppercase letters
    return bool(re.match(r"^\d{4}[A-Z]{3}$", input_string))

def extract_license_plates(frame):
    # Preprocess frame to highlight plate-like contours
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 10, 120)

    # Retrieve all contours for candidate filtering
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    plates = []
    for cnt in contours:
        # Approximate contour to polygon to find rectangular shapes
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ar = w / h
            # Geometric filtering based on aspect ratio and size
            if 2 < ar < 5 and w > 80 and h > 20:
                crop = frame[y:y+h, x:x+w]
                plates.append((x, y, w, h, crop))
    return plates

def extract_license_plate_text(plate_img):
    # Ensure grayscale input for OCR
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img.copy()

    # Enhance characters and binarize image
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # OCR configured for single-line alphanumeric text
    text = pytesseract.image_to_string(
        thresh,
        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    # Clean OCR output and validate format
    clean = "".join(c for c in text if c.isalnum()).upper()
    if len(clean) >= 7:
        candidate = clean[-7:]
        if validar_string(candidate):
            return candidate
    return None


def detectar_matricula_en_frame(frame):
    # Detect and decode first valid license plate in the frame
    plates = extract_license_plates(frame)

    for (x, y, w, h, plate_img) in plates:
        code = extract_license_plate_text(plate_img)
        if code:
            return code, (x, y, w, h)

    return None, None
