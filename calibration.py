import cv2
import numpy as np
import os

def calibrate_camera():
    """
    Tool to calibrate the camera using a chessboard pattern.
    It captures multiple images, finds chessboard corners, and calculates
    camera matrix and distortion coefficients, saving them to .npy files.

    Instructions:
    1. Run this script.
    2. Hold a chessboard in front of the camera, covering different angles and positions.
    3. When the chessboard corners are detected and look good (green lines), press 's' to save the frame.
       - Aim for at least 10-15 good, distinct views (different angles, distances, rotations).
    4. Press 'c' to start calibration once you have enough saved frames.
    5. Press 'q' to quit at any time.
    """
    # Define the chessboard dimensions (number of inner corners per row/column)
    # For an 8x8 chessboard, there are 7x7 inner corners.
    # Adjust these values if your chessboard has a different number of squares.
    chessboard_pattern_size = (7, 7) # (cols, rows) of inner corners
    square_size = 20.0 # Size of one square on your chessboard in a real-world unit (e.g., mm or inches)
                       # This value is only needed if you want to calculate real-world positions.
                       # For just undistorting and perspective transform, its specific value isn't critical.

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (6,6,0)
    # These are the known 3D coordinates of the inner corners in your chessboard's coordinate system.
    objp = np.zeros((chessboard_pattern_size[0] * chessboard_pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_pattern_size[0], 0:chessboard_pattern_size[1]].T.reshape(-1, 2) * square_size

    # Arrays to store object points and image points from all images
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video camera for calibration.")
        return

    print("Camera Calibration Tool started. Press 's' to save frame, 'c' to calibrate, 'q' to quit.")
    print(f"Looking for a {chessboard_pattern_size[0]}x{chessboard_pattern_size[1]} chessboard pattern.")

    saved_frame_count = 0
    img_size = None # Will store (width, height) of frames

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera. Exiting...")
            break

        if img_size is None:
            img_size = (frame.shape[1], frame.shape[0]) # (width, height)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        # Added flags for better detection
        found, corners = cv2.findChessboardCorners(gray, chessboard_pattern_size, None,
                                                    cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                    cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                    cv2.CALIB_CB_FAST_CHECK)

        # If corners are found, draw them
        if found:
            # Refine corner locations for higher accuracy
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            
            # Draw and display the corners
            cv2.drawChessboardCorners(frame, chessboard_pattern_size, corners_refined, found)
            
            # Add text indicating corners found
            cv2.putText(frame, "Corners Found!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No Chessboard Found...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


        cv2.putText(frame, f"Saved frames: {saved_frame_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Calibration Feed (Press s to save, c to calibrate)', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if found:
                objpoints.append(objp)
                imgpoints.append(corners_refined)
                saved_frame_count += 1
                print(f"Frame saved! Total: {saved_frame_count}")
            else:
                print("Cannot save frame: Chessboard not found.")
        elif key == ord('c'):
            if saved_frame_count >= 10: # Recommended minimum 10 frames for calibration
                print("\nStarting calibration...")
                try:
                    # Calibrate the camera
                    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                        objpoints, imgpoints, img_size, None, None
                    )

                    if ret:
                        print("Calibration successful!")
                        print("Camera Matrix:\n", camera_matrix)
                        print("Distortion Coefficients:\n", dist_coeffs)

                        # Save the calibration results
                        np.save("camera_matrix.npy", camera_matrix)
                        np.save("dist_coeffs.npy", dist_coeffs)
                        print("\nCalibration parameters saved to camera_matrix.npy and dist_coeffs.npy")
                        print("You can now use these files in your live feed program to undistort the image.")
                        break # Exit after successful calibration
                    else:
                        print("Calibration failed. Please try again with more or better quality frames.")
                except Exception as e:
                    print(f"An error occurred during calibration: {e}")
                    print("Please ensure you have enough good frames (at least 10-15) and your chessboard pattern size is correct.")
            else:
                print(f"Need more frames for calibration! Saved: {saved_frame_count}. Aim for at least 10-15.")
        elif key == ord('q'):
            print("Exiting calibration tool.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Calibration tool finished.")

if __name__ == "__main__":
    calibrate_camera()
