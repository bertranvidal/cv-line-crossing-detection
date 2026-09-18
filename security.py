from __future__ import annotations

from plate_detector import detectar_matricula_en_frame


def normalize_plate(value: str) -> str:
    """Normalize user-provided plate text before comparison."""
    return "".join(character for character in value if character.isalnum()).upper()


def check_security(frame, authorized_plate: str):
    """Compare the detected plate with the configured authorized plate."""
    detected_plate, bbox = detectar_matricula_en_frame(frame)
    if detected_plate is None:
        return False, None, None

    is_authorized = normalize_plate(detected_plate) == normalize_plate(authorized_plate)
    return is_authorized, detected_plate, bbox
