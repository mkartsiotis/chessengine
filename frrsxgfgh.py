import cv2
import numpy as np

# Global variable to store the reference image of an empty square
# Initialize to None, it will be set when the user presses 'c'
reference_empty_square_image = None
detection_threshold = 10000 # >>> ADJUSTED THRESHOLD BASED ON YOUR FEEDBACK <<<

def capture_live_video_feed():
    """
    Opens the default camera and displays a live video feed.
    It converts the feed to black and white (grayscale).
    It attempts to detect a chessboard, draw its grid, and
    highlight a specific square. It includes a basic
    implementation for detecting a chess piece in the highlighted square
    by comparing it to a captured "empty" reference image.
    Blurring is applied to images before comparison to reduce noise.

    Press 'q' to quit the video feed.
    Press 'c' to capture the current highlighted square as the "empty" reference.
    """
    global reference_empty_square_image, detection_threshold # Declare global variables

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video camera.")
        return

    print("Camera opened successfully. Press 'q' to quit.")
    print("Looking for a chessboard... (try showing a chessboard to the camera!)")
    print("\n--- INSTRUCTIONS FOR PIECE DETECTION ---")
    print("1. Ensure your camera and chessboard are STEADY and LIGHTING is CONSISTENT.")
    print("2. Make sure the HIGHLIGHTED SQUARE is completely EMPTY.")
    print("3. Press the 'c' key to capture the EMPTY SQUARE REFERENCE (it will average 10 frames).")
    print("4. Now, place a chess piece on the HIGHLIGHTED SQUARE.")
    print("5. Observe the 'Diff Sum' values printed in your console:")
    print("   - When EMPTY, the 'Diff Sum' will be low (e.g., around 2000 for your setup).")
    print("   - When a PIECE is present, the 'Diff Sum' will be high (e.g., around 23,000 for your setup).")
    print("6. Adjust the 'detection_threshold' value in the code (around line 6) so it's:")
    print("   - HIGHER than the 'Diff Sum' when EMPTY.")
    print("   - LOWER than the 'Diff Sum' when a PIECE is present.")
    print("   Save the code and restart the script after changing the threshold.")
    print("\n--- DEBUGGING HINTS ---")
    print("   If the highlighted square is not visible, check the 'selected_square_coords' and 'Square Region Shape' in console.")
    print("   'Square Region Shape (0, 0)' indicates a problem with square extraction.")


    # Define the dimensions of the chessboard (number of inner corners per row/column)
    # For an 8x8 chessboard, there are 7x7 inner corners.
    # Adjust these values if your chessboard has a different number of squares.
    chessboard_pattern_size = (7, 7) # (cols, rows) of inner corners

    # Initialize a variable to keep track of the selected square's coordinates
    selected_square_coords = None
    capture_feedback_timer = 0 # Timer for visual feedback after 'c' press
    capture_feedback_duration = 30 # Number of frames to show feedback (approx 1 sec at 30 FPS)
    num_frames_for_reference = 10 # Number of frames to average for reference image capture


    # Define a kernel size for blurring (must be odd numbers)
    blur_kernel_size = (9, 9) # Increased blur for more stability

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame from camera. Exiting...")
            break

        # Convert to Black and White (Grayscale)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        display_frame = frame.copy() # Use a copy to draw on for display

        # Try to Detect Chessboard and its Squares with improved flags
        found, corners = cv2.findChessboardCorners(gray_frame, chessboard_pattern_size, 
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)

        piece_detection_status = "No Board Detected"

        if found:
            corners_2d = corners.reshape(-1, 2)

            # --- Draw grid lines for all squares ---
            for r in range(chessboard_pattern_size[1]):
                for c in range(chessboard_pattern_size[0]):
                    current_corner_index = r * chessboard_pattern_size[0] + c
                    if c < chessboard_pattern_size[0] - 1:
                        p1 = tuple(int(val) for val in corners_2d[current_corner_index])
                        p2 = tuple(int(val) for val in corners_2d[current_corner_index + 1])
                        cv2.line(display_frame, p1, p2, (0, 255, 0), 1)
                    if r < chessboard_pattern_size[1] - 1:
                        p1 = tuple(int(val) for val in corners_2d[current_corner_index])
                        p2 = tuple(int(val) for val in corners_2d[current_corner_index + chessboard_pattern_size[0]])
                        cv2.line(display_frame, p1, p2, (0, 255, 0), 1)
            
            # --- Highlight a specific square (the first fully enclosed internal square) ---
            # This square is defined by the first four inner corners detected:
            # corners[0], corners[1], corners[cols_pattern_size], corners[cols_pattern_size + 1]
            
            p_top_left_of_square = corners_2d[0]
            p_top_right_of_square = corners_2d[1]
            p_bottom_left_of_square = corners_2d[chessboard_pattern_size[0]]
            p_bottom_right_of_square = corners_2d[chessboard_pattern_size[0] + 1]

            # Determine the min/max x and y coordinates from these four points
            all_x_coords = [p_top_left_of_square[0], p_top_right_of_square[0],
                            p_bottom_left_of_square[0], p_bottom_right_of_square[0]]
            all_y_coords = [p_top_left_of_square[1], p_top_right_of_square[1],
                            p_bottom_left_of_square[1], p_bottom_right_of_square[1]]

            estimated_first_square_x1 = int(min(all_x_coords))
            estimated_first_square_y1 = int(min(all_y_coords))
            estimated_first_square_x2 = int(max(all_x_coords))
            estimated_first_square_y2 = int(max(all_y_coords))

            # Ensure coordinates are within frame boundaries and form a valid rectangle
            estimated_first_square_x1 = max(0, estimated_first_square_x1)
            estimated_first_square_y1 = max(0, estimated_first_square_y1)
            estimated_first_square_x2 = min(frame.shape[1], estimated_first_square_x2)
            estimated_first_square_y2 = min(frame.shape[0], estimated_first_square_y2)

            selected_square_coords = (estimated_first_square_x1, estimated_first_square_y1, estimated_first_square_x2, estimated_first_square_y2)

            # Debugging print statement for selected_square_coords
            # print(f"Selected Square Coords: {selected_square_coords}")


            if (selected_square_coords[2] > selected_square_coords[0] and # Check if width > 0
                selected_square_coords[3] > selected_square_coords[1]):    # Check if height > 0
                
                # Determine border color: blue for capture feedback, red otherwise
                border_color = (255, 0, 0) if capture_feedback_timer > 0 else (0, 0, 255)
                cv2.rectangle(display_frame,
                              (selected_square_coords[0], selected_square_coords[1]),
                              (selected_square_coords[2], selected_square_coords[3]),
                              border_color, 3) # Thicker border
                
                cv2.putText(display_frame, 'Checking this square!',
                            (selected_square_coords[0], selected_square_coords[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 1, cv2.LINE_AA)

                # --- Chess Piece Detection in the highlighted square ---
                # Extract square region (only if dimensions are valid)
                square_region_live = gray_frame[selected_square_coords[1]:selected_square_coords[3],
                                                selected_square_coords[0]:selected_square_coords[2]]
                
                # Debugging print statement for square_region_live shape
                print(f"Square Region Shape: {square_region_live.shape}")


                # Apply Gaussian blur to the live square region
                square_region_live = cv2.GaussianBlur(square_region_live, blur_kernel_size, 0)


                if reference_empty_square_image is not None:
                    # Ensure the dimensions match before comparison
                    if square_region_live.shape == reference_empty_square_image.shape and \
                       square_region_live.shape[0] > 0 and square_region_live.shape[1] > 0: # Ensure not empty
                        
                        # Calculate the absolute difference between the live square and the empty reference
                        difference = cv2.absdiff(square_region_live, reference_empty_square_image)
                        
                        # Sum all pixel differences (a high sum indicates a significant change/piece)
                        diff_sum = np.sum(difference)
                        print(f"Diff Sum: {diff_sum}") # <<< ALWAYS PRINT DIFF SUM FOR TUNING >>>

                        # Compare with threshold
                        if diff_sum > detection_threshold:
                            piece_detection_status = "Piece Present!"
                            text_color = (0, 255, 0) # Green for piece
                        else:
                            piece_detection_status = "Empty"
                            text_color = (0, 255, 255) # Yellow for empty
                        
                    else:
                        piece_detection_status = "Reference mismatch / Empty Square"
                        text_color = (0, 165, 255) # Orange for mismatch
                        # print("DEBUG: Reference or live square is empty or dimensions mismatch.")
                else:
                    piece_detection_status = "Press 'c' to set empty ref"
                    text_color = (255, 255, 0) # Cyan for no ref
            else:
                selected_square_coords = None # Reset if invalid
                piece_detection_status = "Invalid Square Area"
                text_color = (0, 0, 255) # Red for invalid square
        else:
            selected_square_coords = None
            piece_detection_status = "No Board Detected"
            text_color = (0, 0, 255) # Red for no board

        # Decrement feedback timer if active
        if capture_feedback_timer > 0:
            capture_feedback_timer -= 1


        # Display piece detection status on the main feed
        cv2.putText(display_frame, piece_detection_status,
                    (10, 30), # Top-left corner for text
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)

        # Display the camera feed (with detections and grid)
        cv2.imshow('Camera Feed (with Chessboard Grid)', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if found and selected_square_coords and \
               selected_square_coords[2] > selected_square_coords[0] and \
               selected_square_coords[3] > selected_square_coords[1]: # Ensure valid square area
                
                print(f"Capturing {num_frames_for_reference} frames for empty reference...")
                temp_reference_frames = []
                for _ in range(num_frames_for_reference):
                    ret_ref, frame_ref = cap.read()
                    if ret_ref:
                        gray_ref = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
                        # Extract the square region from the reference frame
                        square_ref_region = gray_ref[selected_square_coords[1]:selected_square_coords[3],
                                                     selected_square_coords[0]:selected_square_coords[2]]
                        if square_ref_region.shape[0] > 0 and square_ref_region.shape[1] > 0:
                            temp_reference_frames.append(cv2.GaussianBlur(square_ref_region, blur_kernel_size, 0))
                        else:
                            print("Warning: Captured reference frame's square region was empty or invalid.")
                            break
                    else:
                        print("Error: Could not read frame during reference capture.")
                        break
                
                if temp_reference_frames:
                    # Average the captured frames
                    reference_empty_square_image = np.mean(temp_reference_frames, axis=0).astype(np.uint8)
                    print("Reference empty square captured and averaged!")
                    capture_feedback_timer = capture_feedback_duration # Start feedback timer
                else:
                    print("Failed to capture valid reference frames.")

            else:
                print("Cannot capture reference: Chessboard not detected or square invalid/too small.")


    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed.")

if __name__ == "__main__":
    capture_live_video_feed()
