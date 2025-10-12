import cv2

def capture_live_video_feed():
    """
    Opens the default camera and displays a live video feed.
    Press 'q' to quit the video feed.
    """
    # Open the default camera (usually camera index 0)
    # If you have multiple cameras, you might need to try different indices (1, 2, etc.)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video camera.")
        return

    print("Camera opened successfully. Press 'q' to quit.")

    while True:
        # Read a frame from the camera
        # ret: A boolean value, True if the frame was read successfully, False otherwise.
        # frame: The captured frame as a NumPy array.
        ret, frame = cap.read()

        # If the frame was not read successfully, break the loop
        if not ret:
            print("Error: Could not read frame from camera. Exiting...")
            break

        # Display the captured frame in a window
        cv2.imshow('Live Camera Feed (Press "q" to quit)', frame)

        # Wait for a key press for 1 millisecond
        # If 'q' is pressed, break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed.")

if __name__ == "__main__":
    capture_live_video_feed()