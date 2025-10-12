import cv2
import numpy as np

def capture_live_video_feed():
    """
    Opens the default camera and displays a live video feed.
    It converts the feed to black and white (grayscale).
    It attempts to detect a chessboard, draw its grid, and
    highlight a specific square. Comments explain how you could
    conceptually check for a chess piece in that square.
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

    # Initialize a variable to keep track of the selected square's coordinates
    # This will be updated once a chessboard is found
    selected_square_coords = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame from camera. Exiting...")
            break

        # --- 1. Convert to Black and White (Grayscale) ---
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Black and White Feed', gray_frame)

        # --- 2. Try to Detect Chessboard and its Squares ---
        # Find the chessboard corners
        # `found` is True if corners are found, False otherwise
        # `corners` is an array of corner points
        found, corners = cv2.findChessboardCorners(gray_frame, chessboard_pattern_size, None)

        if found:
            # Draw the detected corners on the original color frame for visualization
            cv2.drawChessboardCorners(frame, chessboard_pattern_size, corners, found)

            # Reshape corners to a 2D array of (x, y) points for easier indexing
            corners_2d = corners.reshape(-1, 2)

            # --- Draw grid lines for all squares ---
            # Iterate through rows of inner corners
            for r in range(chessboard_pattern_size[1]):
                # Iterate through columns of inner corners
                for c in range(chessboard_pattern_size[0]):
                    current_corner_index = r * chessboard_pattern_size[0] + c
                    
                    # Draw horizontal lines between corners in the same row
                    if c < chessboard_pattern_size[0] - 1:
                        p1 = tuple(np.int8(corners_2d[current_corner_index]))
                        p2 = tuple(np.int8(corners_2d[current_corner_index + 1]))
                        cv2.line(frame, p1, p2, (0, 255, 0), 1) # Green line

                    # Draw vertical lines between corners in the same column
                    if r < chessboard_pattern_size[1] - 1:
                        p1 = tuple(np.int8(corners_2d[current_corner_index]))
                        p2 = tuple(np.int8(corners_2d[current_corner_index + chessboard_pattern_size[0]]))
                        cv2.line(frame, p1, p2, (0, 255, 0), 1) # Green line
            
            # --- Highlight a specific square (e.g., the top-left internal square) ---
            # This square is formed by corners[0], corners[1], corners[cols_pattern_size], corners[cols_pattern_size + 1]
            # We'll use the first corner (corners[0]) and the one to its right (corners[1])
            # and the one below it (corners[chessboard_pattern_size[0]]) to define the approximate boundaries.
            
            # For demonstration, let's highlight the square 'behind' the first internal corner
            # This is a conceptual approximation. For precise square identification,
            # camera calibration and perspective transformation are usually needed.

            # Get the coordinates of the first inner corner
            p_top_left_inner = corners_2d[0]
            
            # Get the coordinate of the inner corner to its right (first in row 0, col 1)
            p_top_right_inner = corners_2d[1]

            # Get the coordinate of the inner corner below it (first in row 1, col 0)
            p_bottom_left_inner = corners_2d[chessboard_pattern_size[0]]

            # Estimate the size of one square based on the distance between corners
            # This is a rough estimation and works best for a relatively flat board
            square_width_approx = np.linalg.norm(p_top_left_inner - p_top_right_inner)
            square_height_approx = np.linalg.norm(p_top_left_inner - p_bottom_left_inner)

            # Now, let's define the top-left square (conceptually A8 or H1, depending on orientation)
            # We'll assume the board is oriented such that the first inner corner (corners[0])
            # is the top-left of the 8x8 grid of inner corners.
            # The actual square we want to highlight is 'above' and 'left' of corners[0].
            
            # For simplicity and visual demonstration, let's define a bounding box for the
            # square formed by corners[0], corners[1], corners[M], corners[M+1] (where M is pattern_cols)
            # This corresponds to one of the internal 7x7 squares of the corner grid.

            # We'll highlight the square that has corners[0] as its top-left point.
            # To do this, we need the coordinates of the actual *corners of the square*.
            # Let's approximate one square's region using the first few detected inner corners.
            
            # The top-left corner of the *first* internal square (the one between corners[0], corners[1], etc.)
            # would be corners[0].
            # The bottom-right corner of this same square would be corners[chessboard_pattern_size[0] + 1].

            # Let's highlight the first full square defined by internal corners:
            # The square between corners[0], corners[1], corners[cols], corners[cols+1]
            square_top_left_x = int(corners_2d[0][0])
            square_top_left_y = int(corners_2d[0][1])
            
            # The bottom-right corner of this particular square
            square_bottom_right_x = int(corners_2d[1][0]) # Using the x of the next corner horizontally
            square_bottom_right_y = int(corners_2d[chessboard_pattern_size[0]][1]) # Using the y of the next corner vertically

            # Calculate the average width and height of a square based on the pattern size
            # This helps to extend the boundaries for the "first" square on the board's edge
            avg_square_width = np.mean(np.diff(corners_2d[0:chessboard_pattern_size[0], 0])) if chessboard_pattern_size[0] > 1 else square_width_approx
            avg_square_height = np.mean(np.diff(corners_2d[::chessboard_pattern_size[0], 1])) if chessboard_pattern_size[1] > 1 else square_height_approx

            # Let's approximate the *very first square* on the board (e.g., A1/H8)
            # This would be to the 'top-left' of the first detected inner corner (corners_2d[0])
            # This estimation is tricky without full calibration, but for a visual demo:
            
            # Estimated top-left of the actual board's first square (outside inner corners)
            estimated_first_square_x1 = int(corners_2d[0][0] - avg_square_width)
            estimated_first_square_y1 = int(corners_2d[0][1] - avg_square_height)

            # Estimated bottom-right of the actual board's first square
            estimated_first_square_x2 = int(corners_2d[0][0])
            estimated_first_square_y2 = int(corners_2d[0][1])

            # Ensure coordinates are within frame boundaries
            estimated_first_square_x1 = max(0, estimated_first_square_x1)
            estimated_first_square_y1 = max(0, estimated_first_square_y1)
            estimated_first_square_x2 = min(frame.shape[1], estimated_first_square_x2)
            estimated_first_square_y2 = min(frame.shape[0], estimated_first_square_y2)

            selected_square_coords = (estimated_first_square_x1, estimated_first_square_y1, estimated_first_square_x2, estimated_first_square_y2)

            # Draw a thicker rectangle around the highlighted square
            if all(coord >= 0 for coord in selected_square_coords) and \
               selected_square_coords[2] > selected_square_coords[0] and \
               selected_square_coords[3] > selected_square_coords[1]:
                cv2.rectangle(frame,
                              (selected_square_coords[0], selected_square_coords[1]),
                              (selected_square_coords[2], selected_square_coords[3]),
                              (0, 0, 255), 3) # Red, thicker border
                
                # Add text label for the highlighted square
                cv2.putText(frame, 'Checking this square!',
                            (selected_square_coords[0], selected_square_coords[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # --- Conceptual Chess Piece Detection in the highlighted square ---
                # To identify a chess piece in this square, you would typically:
                # 1. Capture an "empty" reference image of this specific square.
                # 2. In live feed, extract the region of this square.
                # 3. Compare the live region with the empty reference.
                #    - Methods:
                #      a. **Simple Pixel Difference:** Calculate the sum of absolute differences
                #         between the live square and the empty reference. If the difference
                #         exceeds a threshold, a change (and thus possibly a piece) is detected.
                #      b. **Color/Brightness Analysis:** Calculate average color or brightness.
                #         A piece will significantly alter these values compared to an empty square.
                #      c. **Contour Detection:** Look for new contours or shapes appearing
                #         within the square's region that are not present in the empty state.
                #      d. **Machine Learning (More Advanced):** Train a simple classifier
                #         (e.g., using a few image features or even a small neural network)
                #         to distinguish "empty" from "occupied".

                # Example (conceptual, not actual detection logic implemented here):
                # if selected_square_coords:
                #     square_region = frame[selected_square_coords[1]:selected_square_coords[3],
                #                           selected_square_coords[0]:selected_square_coords[2]]
                #     # You would then analyze 'square_region' for a piece
                #     # For instance, if you had a reference_empty_square_image:
                #     # diff = cv2.absdiff(square_region, reference_empty_square_image)
                #     # if np.sum(diff) > some_threshold:
                #     #     print("Piece detected in highlighted square!")
                #     # else:
                #     #     print("Highlighted square appears empty.")
        else:
            # If no chessboard is found, reset the selected square coordinates
            selected_square_coords = None

        # Display the original camera feed (with detections and grid)
        cv2.imshow('Camera Feed (with Chessboard Grid)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed.")

if __name__ == "__main__":
    capture_live_video_feed()
