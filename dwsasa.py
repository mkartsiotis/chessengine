import cv2
import numpy as np

def capture_live_video_feed():
    """
    Opens the default camera and displays a live video feed.
    It converts the feed to black and white (grayscale) and
    attempts to detect and crop a chessboard if found.
    Press 'q' to quit the video feed.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video camera.")
        return

    print("Camera opened successfully. Press 'q' to quit.")
    print("Looking for a chessboard... (try showing a chessboard to the camera!)")

    # Define the dimensions of the chessboard (number of inner corners per row/column)
    # For an 8x8 chessboard, there are 7x7 inner corners.
    # Adjust these values if your chessboard has a different number of squares.
    chessboard_pattern_size = (7, 7) # (cols, rows) of inner corners

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame from camera. Exiting...")
            break

        # --- 1. Convert to Black and White (Grayscale) ---
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Black and White Feed', gray_frame)

        # --- 2. Try to Detect and Crop Chessboard ---
        # Find the chessboard corners
        # `found` is True if corners are found, False otherwise
        # `corners` is an array of corner points
        found, corners = cv2.findChessboardCorners(gray_frame, chessboard_pattern_size, None)

        if found:
            # Draw the detected corners on the original color frame for visualization
            # This helps to see what the camera is detecting as a chessboard
            cv2.drawChessboardCorners(frame, chessboard_pattern_size, corners, found)

            # Calculate the bounding box around the detected corners
            # Reshape corners to a 2D array of (x, y) points
            corners_2d = corners.reshape(-1, 2)
            x_min, y_min = np.int8(corners_2d.min(axis=0))
            x_max, y_max = np.int8(corners_2d.max(axis=0))

            # Add a little padding to the bounding box
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(frame.shape[1], x_max + padding)
            y_max = min(frame.shape[0], y_max + padding)

            # Crop the original color frame using the calculated bounding box
            cropped_chessboard = frame[y_min:y_max, x_min:x_max]

            # Display the cropped chessboard
            if cropped_chessboard.shape[0] > 0 and cropped_chessboard.shape[1] > 0:
                cv2.imshow('Cropped Chessboard', cropped_chessboard)
            else:
                print("Cropped chessboard area is too small, skipping display.")
        else:
            # If no chessboard is found, ensure the "Cropped Chessboard" window is closed
            # or not shown, to avoid displaying an old cropped image.
            # You might want to add a blank image or close the window if it exists.
            try:
                cv2.destroyWindow('Cropped Chessboard')
            except cv2.error:
                pass # Window might not exist yet

        # Display the original (now with drawn corners if found) frame
        cv2.imshow('Original Camera Feed (with detections)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed.")

if __name__ == "__main__":
    capture_live_video_feed()
