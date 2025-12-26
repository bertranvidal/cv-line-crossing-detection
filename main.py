import cv2
import time

from security import check_security
from line_detector import detectar_linea_continua
from car_detector import detectar_coche
from tracker import KalmanTracker
import drawer



# Parameters

MARGIN_LINEA = 30        # Pixel margin around the line considered as neutral zone
FRAMES_ESTABLE = 8       # Consecutive frames required to confirm a stable side
POST_CRUCE_TIME = 3.0   # Seconds to keep recording after crossing detection


def side_or_none(center, linea):
    """
    Returns relative position of a point with respect to a line:
    LEFT, RIGHT, UP, DOWN or None if inside margin zone
    """
    x1, y1, x2, y2 = linea

    # Predominantly horizontal line
    if abs(x2 - x1) > abs(y2 - y1):
        y_line = (y1 + y2) // 2
        dy = center[1] - y_line
        if abs(dy) < MARGIN_LINEA:
            return None
        return "UP" if dy < 0 else "DOWN"

    # Predominantly vertical line
    else:
        x_line = (x1 + x2) // 2
        dx = center[0] - x_line
        if abs(dx) < MARGIN_LINEA:
            return None
        return "LEFT" if dx < 0 else "RIGHT"


def main():
    # Video capture and tracker initialization
    cap = cv2.VideoCapture(0)
    tracker = KalmanTracker()

    # Video writer for full recording
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        'grabacion_completa.avi',
        fourcc,
        20.0,
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
    )

    linea = None
    acceso = False

    lado_inicial = None
    contador_lado = 0
    ultimo_lado = None

    cruce_detectado = False
    t_cruce = None

    matricula_detectada = None
    bbox_matricula = None
    tiempo_espera = 2.0
    t_acceso = None

    frame_count = 0

    # FPS variables
    t_prev = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Real time FPS estimation
        t_now = time.time()
        dt = t_now - t_prev
        t_prev = t_now
        if dt > 0:
            fps = 1.0 / dt

        frame_count += 1

      
        # PHASE 1: SECURITY CHECK
    
        if not acceso:
            if frame_count % 10 == 0:
                ok, mat, bbox = check_security(frame)
                if mat is not None and bbox is not None:
                    matricula_detectada = mat
                    bbox_matricula = bbox
                if ok:
                    acceso = True
                    t_acceso = time.time()

            if bbox_matricula is not None:
                x, y, w, h = bbox_matricula
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, matricula_detectada, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            cv2.putText(frame, "Scanning license plate...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            drawer.draw_fps(frame, fps)
            out.write(frame)
            cv2.imshow("Sistema", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        
        # WAIT AFTER ACCESS GRANTED 
       
        if time.time() - t_acceso < tiempo_espera:
            cv2.putText(frame, "ACCESS GRANTED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            drawer.draw_fps(frame, fps)
            out.write(frame)
            cv2.imshow("Sistema", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        
        # LINE DETECTION (ONCE)
        
        if linea is None:
            linea = detectar_linea_continua(frame)

        
        # VEHICLE DETECTION + TRACKING
        
        bbox, center = detectar_coche(frame)
        pred_center = None
        if center is not None:
            pred_center = tracker.update(center)

        
        # INITIAL SIDE STABILIZATION
       
        if linea is not None and pred_center is not None and lado_inicial is None:
            lado = side_or_none(pred_center, linea)

            if lado is None:
                contador_lado = 0
                ultimo_lado = None
            else:
                if lado == ultimo_lado:
                    contador_lado += 1
                else:
                    ultimo_lado = lado
                    contador_lado = 1

                if contador_lado >= FRAMES_ESTABLE:
                    lado_inicial = lado

        
        # LINE CROSSING DETECTION
        
        if lado_inicial is not None and pred_center is not None and not cruce_detectado:
            lado_actual = side_or_none(pred_center, linea)
            if lado_actual is not None and lado_actual != lado_inicial:
                cruce_detectado = True
                t_cruce = time.time()
                print("CONTINUOUS LINE CROSSING DETECTED")

       
        # DRAWING AND OUTPUT
        
        drawer.draw_line(frame, linea)
        drawer.draw_car(frame, bbox, pred_center)

        if cruce_detectado:
            text = "CONTINUOUS LINE CROSSING"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1.0
            thickness = 3
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            h, w, _ = frame.shape
            cv2.putText(frame, text, ((w - tw)//2, (h + th)//2),
                        font, scale, (0, 0, 255), thickness)

        drawer.draw_fps(frame, fps)
        out.write(frame)
        cv2.imshow("Sistema", frame)

        if cruce_detectado and time.time() - t_cruce > POST_CRUCE_TIME:
            break

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
