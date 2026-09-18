from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

import drawer
from car_detector import detectar_coche
from line_detector import detectar_linea_continua
from plate_detector import configure_tesseract
from security import check_security
from tracker import KalmanTracker


LINE_MARGIN = 30
STABLE_FRAMES = 8
POST_CROSSING_SECONDS = 3.0
ACCESS_DELAY_SECONDS = 2.0


def parse_source(value: str) -> int | str:
    """Interpret a numeric source as a camera index and any other value as a path."""
    return int(value) if value.isdigit() else value


def side_or_none(center, line):
    """Return the stable side of a point relative to a detected line."""
    x1, y1, x2, y2 = line

    if abs(x2 - x1) > abs(y2 - y1):
        line_y = (y1 + y2) // 2
        delta = center[1] - line_y
        if abs(delta) < LINE_MARGIN:
            return None
        return "UP" if delta < 0 else "DOWN"

    line_x = (x1 + x2) // 2
    delta = center[0] - line_x
    if abs(delta) < LINE_MARGIN:
        return None
    return "LEFT" if delta < 0 else "RIGHT"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect vehicle crossings over a continuous line.",
    )
    parser.add_argument("--source", default="0", help="Camera index or input video path.")
    parser.add_argument(
        "--output",
        default="outputs/line_crossing_output.avi",
        help="Path for the annotated output video.",
    )
    parser.add_argument(
        "--authorized-plate",
        default="",
        help="Authorized Spanish plate. Required unless --skip-security is used.",
    )
    parser.add_argument(
        "--skip-security",
        action="store_true",
        help="Skip plate access control and start line detection immediately.",
    )
    parser.add_argument(
        "--tesseract-cmd",
        default="",
        help="Optional path to the Tesseract executable.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    if not args.skip_security and not args.authorized_plate:
        raise ValueError(
            "Provide --authorized-plate or use --skip-security for the line-crossing demo."
        )

    if not args.skip_security:
        configure_tesseract(args.tesseract_cmd or None)

    source = parse_source(args.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    output_fps = source_fps if source_fps and source_fps > 1 else 20.0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"XVID"),
        output_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    tracker = KalmanTracker()
    line = None
    access_granted = args.skip_security
    access_time = time.time() - ACCESS_DELAY_SECONDS if access_granted else None

    initial_side = None
    stable_side_count = 0
    last_side = None
    crossing_detected = False
    crossing_time = None
    detected_plate = None
    plate_bbox = None
    frame_count = 0
    previous_time = time.time()
    fps = 0.0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            now = time.time()
            elapsed = now - previous_time
            previous_time = now
            if elapsed > 0:
                fps = 1.0 / elapsed

            frame_count += 1

            if not access_granted:
                if frame_count % 10 == 0:
                    is_authorized, plate, bbox = check_security(
                        frame,
                        args.authorized_plate,
                    )
                    if plate is not None and bbox is not None:
                        detected_plate = plate
                        plate_bbox = bbox
                    if is_authorized:
                        access_granted = True
                        access_time = time.time()

                if plate_bbox is not None:
                    x, y, box_width, box_height = plate_bbox
                    color = (0, 255, 0) if access_granted else (0, 0, 255)
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + box_width, y + box_height),
                        color,
                        3,
                    )
                    cv2.putText(
                        frame,
                        detected_plate,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        color,
                        3,
                    )

                cv2.putText(
                    frame,
                    "Scanning license plate...",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                )
                drawer.draw_fps(frame, fps)
                writer.write(frame)
                cv2.imshow("Line crossing detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            if access_time is not None and time.time() - access_time < ACCESS_DELAY_SECONDS:
                cv2.putText(
                    frame,
                    "ACCESS GRANTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    3,
                )
                drawer.draw_fps(frame, fps)
                writer.write(frame)
                cv2.imshow("Line crossing detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            if line is None:
                line = detectar_linea_continua(frame)

            bbox, center = detectar_coche(frame)
            predicted_center = tracker.update(center) if center is not None else None

            if line is not None and predicted_center is not None and initial_side is None:
                current_side = side_or_none(predicted_center, line)
                if current_side is None:
                    stable_side_count = 0
                    last_side = None
                elif current_side == last_side:
                    stable_side_count += 1
                else:
                    last_side = current_side
                    stable_side_count = 1

                if stable_side_count >= STABLE_FRAMES:
                    initial_side = current_side

            if (
                initial_side is not None
                and predicted_center is not None
                and not crossing_detected
            ):
                current_side = side_or_none(predicted_center, line)
                if current_side is not None and current_side != initial_side:
                    crossing_detected = True
                    crossing_time = time.time()
                    print("CONTINUOUS LINE CROSSING DETECTED")

            drawer.draw_line(frame, line)
            drawer.draw_car(frame, bbox, predicted_center)

            if crossing_detected:
                text = "CONTINUOUS LINE CROSSING"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1.0
                thickness = 3
                (text_width, text_height), _ = cv2.getTextSize(
                    text, font, scale, thickness
                )
                frame_height, frame_width, _ = frame.shape
                cv2.putText(
                    frame,
                    text,
                    (
                        (frame_width - text_width) // 2,
                        (frame_height + text_height) // 2,
                    ),
                    font,
                    scale,
                    (0, 0, 255),
                    thickness,
                )

            drawer.draw_fps(frame, fps)
            writer.write(frame)
            cv2.imshow("Line crossing detection", frame)

            if (
                crossing_detected
                and crossing_time is not None
                and time.time() - crossing_time > POST_CROSSING_SECONDS
            ):
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Annotated video saved to: {output_path}")


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
