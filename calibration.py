
from typing import List
import numpy as np
import imageio
import cv2
import copy
import glob


def load_images(filenames: List) -> List:
    # Load a list of images from disk using imageio
    return [imageio.imread(filename) for filename in filenames]

def show_image(img, window_name="Image"):
    # Display an image and block execution until a key is pressed
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def write_image(filename, img):
    # Save image to disk using OpenCV
    cv2.imwrite(filename, img)

def get_chessboard_points(chessboard_shape, dx, dy):
    # Generate 3D coordinates of chessboard corners assuming Z=0 plane
    objp = np.zeros((chessboard_shape[0] * chessboard_shape[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_shape[0], 0:chessboard_shape[1]].T.reshape(-1, 2)
    # Scale points according to real square size
    objp[:, 0] *= dx   
    objp[:, 1] *= dy   
    return objp

if __name__ == "__main__":
    # Load all calibration images from data folder
    imgs_path = sorted(glob.glob("./data/*.jpg"))  
    imgs = load_images(imgs_path)
   
    # Initial chessboard corner detection 
    corners = [cv2.findChessboardCorners(img, (6,9), None) for img in imgs]
   
   
    corners_copy = copy.deepcopy(corners)
    # Termination criteria for subpixel corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    # Convert images to grayscale for subpixel refinement
    imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
    # Refine detected corners to subpixel accuracy
    corners_refined = [
        cv2.cornerSubPix(i, cor[1], (11, 11), (-1, -1), criteria) if cor[0] else []
        for i, cor in zip(imgs_gray, corners_copy)
    ]
    
    
    imgs_copy = copy.deepcopy(imgs)
    
    # Draw refined chessboard corners on images
    for img, cor, det in zip(imgs_copy, corners_refined, corners):
        if len(cor) != 0:
            cv2.drawChessboardCorners(img, (6,9), cor, det[0])
            
    # Visual check of detected corners
    show_image(imgs_copy[1], "Corners - Image ")
    for i, img in enumerate(imgs_copy, start=0):
        write_image(f"corners/corners_image{i}.jpg", img)  # extra

    # Generate corresponding 3D object points only for valid detections
    chessboard_points = [
        get_chessboard_points((6,9), 30, 30)
        for _ in range(len(corners_refined)) if len(corners_refined[_]) != 0
    ]

    # Extract valid 2D image points
    valid_corners = [cor[1] for cor in corners if cor[0]]
    valid_corners = np.asarray(valid_corners, dtype=np.float32)
    
    # Perform camera calibration to obtain intrinsics and distortion
    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        chessboard_points,
        valid_corners,
        imgs_gray[0].shape[::-1],
        None,
        None
    )

    # Build extrinsic matrices for each view
    extrinsics = list(
        map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs)
    )
    
    # Output calibration results
    print("Intrinsics:\n", intrinsics)
    print("Distortion coefficients:\n", dist_coeffs)
    print("Root mean squared reprojection error:\n", rms)
