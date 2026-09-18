from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytesseract


def configure_tesseract(command: str | None = None) -> str:
    """Configure the Tesseract executable from a CLI argument, env var or PATH."""
    candidate = command or os.getenv("TESSERACT_CMD") or shutil.which("tesseract")
    if not candidate:
        raise RuntimeError(
            "Tesseract was not found. Install it and add it to PATH, "
            "or provide --tesseract-cmd / set TESSERACT_CMD."
        )

    resolved = shutil.which(candidate) or candidate
    if not Path(resolved).exists():
        raise FileNotFoundError(f"Tesseract executable not found: {resolved}")

    pytesseract.pytesseract.tesseract_cmd = str(resolved)
    return str(resolved)


def validar_string(input_string: str) -> bool:
    """Validate the current Spanish plate format: four digits and three letters."""
    return bool(re.fullmatch(r"\d{4}[A-Z]{3}", input_string))


def extract_license_plates(frame):
    """Return rectangular regions that resemble a license plate."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 10, 120)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    plates = []

    for contour in contours:
        approx = cv2.approxPolyDP(
            contour,
            0.02 * cv2.arcLength(contour, True),
            True,
        )
        if len(approx) != 4:
            continue

        x, y, width, height = cv2.boundingRect(approx)
        aspect_ratio = width / height
        if 2 < aspect_ratio < 5 and width > 80 and height > 20:
            crop = frame[y : y + height, x : x + width]
            plates.append((x, y, width, height, crop))

    return plates


def extract_license_plate_text(plate_img):
    """Run OCR on a plate crop and return a validated Spanish plate."""
    gray = (
        cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        if len(plate_img.shape) == 3
        else plate_img.copy()
    )

    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    _, threshold = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    text = pytesseract.image_to_string(
        threshold,
        config=(
            "--psm 7 "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ),
    )

    clean = "".join(character for character in text if character.isalnum()).upper()
    if len(clean) >= 7:
        candidate = clean[-7:]
        if validar_string(candidate):
            return candidate
    return None


def detectar_matricula_en_frame(frame):
    """Detect and decode the first valid license plate in a frame."""
    for x, y, width, height, plate_img in extract_license_plates(frame):
        code = extract_license_plate_text(plate_img)
        if code:
            return code, (x, y, width, height)

    return None, None
