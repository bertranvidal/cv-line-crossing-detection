import cv2

def draw_line(frame, line):
    # Draw detected reference line 
    if line:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1,y1), (x2,y2), (0,0,255), 3)

def draw_car(frame, bbox, center):
    # Draw vehicle bounding box and its tracked center
    if bbox:
        x,y,w,h = bbox
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
    if center:
        cx, cy = center
        cv2.circle(frame, (cx,cy), 5, (0,255,0), -1)

def draw_alert(frame):
    # Visual alert when a line crossing is detected
    cv2.putText(frame, "CRUCE DETECTADO", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


def draw_fps(frame, fps):
    # Overlay current frames-per-second for performance monitoring
    text = f"FPS: {int(fps)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    h, w, _ = frame.shape
    x = w - tw - 20
    y = 40
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)