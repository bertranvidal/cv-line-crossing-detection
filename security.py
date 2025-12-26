from plate_detector import detectar_matricula_en_frame
# Plate to detect
MATRICULA_OBJETIVO = "4676NAH"  

def check_security(frame):
    matricula, bbox = detectar_matricula_en_frame(frame)
    # Check if the detected plate is the correct one
    if matricula == MATRICULA_OBJETIVO:
        return True, matricula, bbox

    return False, None, None
