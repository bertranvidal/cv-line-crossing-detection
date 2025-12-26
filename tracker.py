import numpy as np
import cv2

class KalmanTracker:
    def __init__(self):
        # Kalman filter with 4D state (x, y, vx, vy) and 2D measurement (x, y)
        self.kf = cv2.KalmanFilter(4, 2)

        # Measurement model: we directly observe position
        self.kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)

        # Constant-velocity motion model
        self.kf.transitionMatrix = np.array(
            [[1,0,1,0],
             [0,1,0,1],
             [0,0,1,0],
             [0,0,0,1]], np.float32)

        # Process noise to allow smooth but adaptive motion
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2

        self.init = False

    def update(self, measurement):
        # Correct state with measurement when available
        if measurement is not None:
            cx, cy = measurement
            meas = np.array([[np.float32(cx)], [np.float32(cy)]])
            if not self.init:
                # Initialize state on first detection
                self.kf.statePost = np.array([[cx],[cy],[0],[0]], np.float32)
                self.init = True
            self.kf.correct(meas)

        # Predict next state (used even if no measurement is available)
        pred = self.kf.predict()
        return int(pred[0]), int(pred[1])
