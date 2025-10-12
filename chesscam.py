import cv2
import numpy as np
import os
import time
import chess
import chess.engine
import asyncio
import pyttsx3 # Import for text-to-speech

# --- Calibration File Paths ---
CAMERA_MATRIX_FILE = "camera_matrix.npy"
DIST_COEFFS_FILE = "dist_coeffs.npy"

# Global variables for calibration and piece detection
camera_matrix = None
dist_coeffs = None

# --- CONSTANTS FOR PIECE TYPES ---
EMPTY_SQUARE = 0
PLAYER_1_PIECE = 1 # Representing PHYSICAL RED pieces (which are CHESS BLACK)
PLAYER_2_PIECE = 2 # Representing PHYSICAL YELLOW pieces (which are CHESS WHITE)

# --- HSV RANGES FOR TWO COLORS ---
# Red (typical range, might need adjustment for your lighting/camera)
# Note: Red wraps around in HSV, so often needs two ranges
lower_red1 = np.array([0, 70, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 70]) # Second range for wrapping red
upper_red2 = np.array([180, 255, 255])

# Yellow (typical range, **UPDATED WITH YOUR CALIBRATED VALUES**)
lower_yellow = np.array([30, 70, 70])
upper_yellow = np.array([60, 255, 255])

# --- PIXEL THRESHOLDS FOR DETECTION ---
red_pixel_threshold = 15
yellow_pixel_threshold = 90 # Updated as requested

blur_kernel_size = (5, 5)

# --- GLOBAL VARIABLES FOR INTERACTIVE MOVE DETECTION ---
detection_mode = 'waiting_for_start' # States: 'waiting_for_start', 'waiting_for_move_complete'
pre_move_board_state = None
post_move_board_state = None

pending_move_data = {'move': None, 'count': 0}
consecutive_move_confirmations_needed = 2

# --- CHESS ENGINE GLOBAL VARIABLES ---
chess_board = None
chess_engine = None
engine_ready = False
# Human player controls the Yellow (White) pieces by default.
# This means the program expects the first move to be made by a Yellow piece.
player_to_move_color = chess.WHITE

# --- TEXT-TO-SPEECH ENGINE GLOBAL VARIABLE ---
speech_engine = None


def load_calibration_parameters():
    """Loads camera matrix and distortion coefficients from .npy files."""
    global camera_matrix, dist_coeffs
    if os.path.exists(CAMERA_MATRIX_FILE) and os.path.exists(DIST_COEFFS_FILE):
        camera_matrix = np.load(CAMERA_MATRIX_FILE)
        dist_coeffs = np.load(DIST_COEFFS_FILE)
        print(f"Calibration parameters loaded from {CAMERA_MATRIX_FILE} and {DIST_COEFFS_FILE}")
        return True
    else:
        print("Calibration files not found. Please run the 'Camera Calibration Tool' first to generate them.")
        print("Expected files: camera_matrix.npy, dist_coeffs.npy in the same directory.")
        return False

def get_chessboard_notation(row, col):
    """Converts 0-indexed (row, col) to standard chessboard notation (e.g., (7,0) -> 'A1')."""
    # Rows: 0 (rank 8), 1 (rank 7), ..., 7 (rank 1)
    # Cols: 0 (file A), 1 (file B), ..., 7 (file H)
    file_char = chr(ord('A') + col)
    rank_char = str(8 - row)
    return f"{file_char}{rank_char}"

def order_points(pts):
    """
    Orders a list of 4 points such that the first entry is the top-left,
    the second is the top-right, the third is the bottom-right, and the fourth
    is the bottom-left. This is robust to rotation.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-Left
    rect[2] = pts[np.argmax(s)] # Bottom-Right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-Right
    rect[3] = pts[np.argmax(diff)] # Bottom-Left
    return rect


# Make the main function asynchronous
async def capture_live_video_feed():
    """
    Opens the default camera and displays a live video feed.
    It uses loaded calibration parameters to undistort the image
    and then applies perspective transform to get a top-down view of the chessboard.
    It identifies and highlights all 64 squares and detects piece presence
    based on red and yellow color detection.

    Interactive move detection:
    - Press 's' to indicate you are about to start a move (records pre-move state).
    - Make your move.
    - Press 'm' to indicate the move is complete (compares to pre-move state and identifies).
    - Press 'q' to quit.
    """
    global red_pixel_threshold, yellow_pixel_threshold, camera_matrix, dist_coeffs
    global lower_red1, upper_red1, lower_red2, upper_red2, lower_yellow, upper_yellow
    global detection_mode, pre_move_board_state, post_move_board_state
    global pending_move_data
    global chess_board, chess_engine, engine_ready, player_to_move_color
    global speech_engine # Reference the global speech engine

    if not load_calibration_parameters():
        return

    # --- Initialize Chess Engine ---
    # IMPORTANT: REPLACE THIS WITH THE ACTUAL PATH TO YOUR STOCKFISH EXECUTABLE
    STOCKFISH_PATH = r"C:\Users\evaka\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe" # Example for Windows
    # STOCKFISH_PATH = "/usr/local/bin/stockfish" # Example for Linux/macOS
    # STOCKFISH_PATH = "/Users/youruser/Downloads/stockfish/stockfish" # Example macOS path

    if not os.path.exists(STOCKFISH_PATH):
        print(f"Error: Stockfish executable not found at '{STOCKFISH_PATH}'. Please download it and set the correct path.")
        return

    try:
        _transport, chess_engine = await chess.engine.popen_uci(STOCKFISH_PATH)
        
        print(f"DEBUG: Type of chess_engine after popen_uci: {type(chess_engine)}")
        
        chess_board = chess.Board() # Initialize a new chess board
        engine_ready = True
        print("\n--- Chess Engine (Stockfish) initialized. ---")
        print("Initial FEN: ", chess_board.fen())
        print(f"Human player assumed to be: {'Yellow (White)' if player_to_move_color == chess.WHITE else 'Red (Black)'}")

    except Exception as e:
        print(f"Error initializing Stockfish engine: {e}")
        print("Chess engine features will be unavailable.")
        engine_ready = False

    # --- Initialize Text-to-Speech Engine ---
    try:
        speech_engine = pyttsx3.init()
        # Optional: Adjust speech properties (uncomment and modify as needed)
        # voices = speech_engine.getProperty('voices')
        # speech_engine.setProperty('voice', voices[0].id) # Try different indices for other voices
        # speech_engine.setProperty('rate', 150) # Words per minute (default is usually 200)
        # speech_engine.setProperty('volume', 1.0) # Volume (0.0 to 1.0)
        print("Text-to-speech engine initialized.")
        
        # --- Welcome Message (NEWLY ADDED) ---
        welcome_message = "Welcome to the electronic chess board. You will be sure defeated, so if you like to lose, let's play together chess!"
        speech_engine.say(welcome_message)
        speech_engine.runAndWait()
        print(f"Spoke welcome message: \"{welcome_message}\"") # For console confirmation

    except Exception as e:
        print(f"Warning: Could not initialize text-to-speech engine: {e}")
        speech_engine = None # Ensure it's None if initialization fails


    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video camera.")
        # If engine was opened, close it
        if engine_ready:
            await chess_engine.quit() # Use await for quitting
        # Close speech engine if initialized
        if speech_engine:
            speech_engine.stop() # Or engine_speech.endLoop() depending on usage
        return

    print("Camera opened successfully. Press 'q' to quit.")
    print("Looking for a chessboard... (try showing a chessboard to the camera!)")
    print("\n--- IMPORTANT: PIECE COLORING FOR CAPTURE DETECTION ---")
    print("   - Your physical **YELLOW** pieces are **CHESS WHITE**.")
    print("   - Your physical **RED** pieces are **CHESS BLACK**.")
    print(f"   - Adjust `red_pixel_threshold` ({red_pixel_threshold}) and `yellow_pixel_threshold` ({yellow_pixel_threshold}) as needed.")
    print("\n--- INTERACTIVE MOVE DETECTION ---")
    print("   - Press 's' (start) BEFORE you move a piece.")
    print("   - Make your move (and remove your hand).")
    print("   - Press 'm' (move completed) AFTER the piece is moved and board is clear.")


    chessboard_pattern_size = (7, 7)
    expected_corner_count = chessboard_pattern_size[0] * chessboard_pattern_size[1]

    output_board_size = 640
    square_output_size = output_board_size // 8

    target_board_corners = np.float32([
        [0, 0],
        [output_board_size - 1, 0],
        [output_board_size - 1, output_board_size - 1],
        [0, output_board_size - 1]
    ])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera. Exiting...")
            break

        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, camera_matrix)
        gray_undistorted = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray_undistorted, chessboard_pattern_size, None,
                                                    cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                    cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                    cv2.CALIB_CB_FAST_CHECK)

        perspective_display_frame = np.zeros((output_board_size, output_board_size, 3), np.uint8)

        status_text = ""
        text_color = (0, 0, 255) # Red for errors/waiting

        current_board_state_temp = None # Initialize as None, only set if board found

        if found and len(corners) == expected_corner_count:
            corners_refined = cv2.cornerSubPix(gray_undistorted, corners, (11, 11), (-1, -1),
                                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            raw_inner_corners_for_ordering = np.float32([
                corners_refined[0],
                corners_refined[chessboard_pattern_size[0] - 1],
                corners_refined[-1],
                corners_refined[chessboard_pattern_size[0] * (chessboard_pattern_size[1] - 1)]
            ]).reshape(-1, 2)

            ordered_spatial_inner_corners = order_points(raw_inner_corners_for_ordering)

            vec_x_unit = (ordered_spatial_inner_corners[1] - ordered_spatial_inner_corners[0]) / (chessboard_pattern_size[0] - 1)
            vec_y_unit = (ordered_spatial_inner_corners[3] - ordered_spatial_inner_corners[0]) / (chessboard_pattern_size[1] - 1)

            src_outer_corners_normalized = np.float32([
                ordered_spatial_inner_corners[0] - vec_x_unit - vec_y_unit,
                ordered_spatial_inner_corners[1] + vec_x_unit - vec_y_unit,
                ordered_spatial_inner_corners[2] + vec_x_unit + vec_y_unit,
                ordered_spatial_inner_corners[3] - vec_x_unit + vec_y_unit
            ])

            M = cv2.getPerspectiveTransform(src_outer_corners_normalized, target_board_corners)

            undistorted_frame_clean_for_detection = undistorted_frame.copy()
            cv2.drawChessboardCorners(undistorted_frame, chessboard_pattern_size, corners_refined, found)

            warped_clean_frame_for_detection = cv2.warpPerspective(undistorted_frame_clean_for_detection, M, (output_board_size, output_board_size))
            hsv_perspective_frame = cv2.cvtColor(warped_clean_frame_for_detection, cv2.COLOR_BGR2HSV)
            perspective_display_frame = warped_clean_frame_for_detection.copy()

            for i in range(1, 8):
                cv2.line(perspective_display_frame, (0, i * square_output_size), (output_board_size, i * square_output_size), (200, 200, 200), 1)
                cv2.line(perspective_display_frame, (i * square_output_size, 0), (i * square_output_size, output_board_size), (200, 200, 200), 1)

            all_squares_detected_count = 0
            # Initialize current_board_state_temp with EMPTY_SQUARE (0)
            current_board_state_temp = np.full((8, 8), EMPTY_SQUARE, dtype=int)

            for r in range(8):
                for c in range(8):
                    x1 = c * square_output_size
                    y1 = r * square_output_size
                    x2 = (c + 1) * square_output_size
                    y2 = (r + 1) * square_output_size

                    if x2 > x1 and y2 > y1:
                        square_region_live_hsv = hsv_perspective_frame[y1:y2, x1:x2]
                        square_region_live_hsv_blurred = cv2.GaussianBlur(square_region_live_hsv, blur_kernel_size, 0)

                        # --- DETECT RED PIECES ---
                        red_mask1 = cv2.inRange(square_region_live_hsv_blurred, lower_red1, upper_red1)
                        red_mask2 = cv2.inRange(square_region_live_hsv_blurred, lower_red2, upper_red2)
                        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
                        red_pixel_count = np.sum(red_mask) // 255

                        # --- DETECT YELLOW PIECES ---
                        yellow_mask = cv2.inRange(square_region_live_hsv_blurred, lower_yellow, upper_yellow)
                        yellow_pixel_count = np.sum(yellow_mask) // 255

                        if red_pixel_count > red_pixel_threshold:
                            current_board_state_temp[r, c] = PLAYER_1_PIECE
                            overlay = perspective_display_frame.copy()
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1) # Red highlight (BGR)
                            alpha = 0.4
                            perspective_display_frame = cv2.addWeighted(overlay, alpha, perspective_display_frame, 1 - alpha, 0)
                        elif yellow_pixel_count > yellow_pixel_threshold:
                            current_board_state_temp[r, c] = PLAYER_2_PIECE
                            overlay = perspective_display_frame.copy()
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1) # Yellow highlight (BGR)
                            alpha = 0.4
                            perspective_display_frame = cv2.addWeighted(overlay, alpha, perspective_display_frame, 1 - alpha, 0)
                        else:
                            current_board_state_temp[r, c] = EMPTY_SQUARE # No piece detected

                        all_squares_detected_count += 1

            # --- Update Status Text based on detection_mode ---
            if all_squares_detected_count == 64:
                # If board is fully detected and stable
                if detection_mode == 'waiting_for_start':
                    status_text = "Ready. Press 's' to start your move."
                    text_color = (0, 255, 0) # Green for ready
                elif detection_mode == 'waiting_for_move_complete':
                    status_text = "Move in progress? Press 'm' when done."
                    text_color = (0, 165, 255) # Orange for in-progress
            else:
                if detection_mode == 'waiting_for_move_complete':
                    status_text = f"Board obscured ({all_squares_detected_count}/64 squares). Waiting for 'm'."
                    text_color = (0, 0, 255) # Red to indicate current obstruction
                else:
                    status_text = f"Incomplete Board Detection ({all_squares_detected_count}/64 squares)"
                    text_color = (0, 165, 255) # Orange for incomplete
                    detection_mode = 'waiting_for_start'
                    pre_move_board_state = None
                    post_move_board_state = None
                    pending_move_data = {'move': None, 'count': 0}

        else: # No chessboard found at all (found is False)
            if detection_mode == 'waiting_for_move_complete':
                status_text = "Board not visible. Waiting for 'm'."
                text_color = (0, 0, 255) # Red to indicate current obstruction
            else:
                status_text = "No Board Detected. Place chessboard in view."
                text_color = (0, 0, 255) # Red
                detection_mode = 'waiting_for_start'
                pre_move_board_state = None
                post_move_board_state = None
                pending_move_data = {'move': None, 'count': 0}


        # --- Display Status Text ---
        cv2.putText(undistorted_frame, status_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)

        # Display current player's turn if engine is ready
        if engine_ready:
            # Map chess.WHITE/BLACK to your physical colors for display
            display_turn_color = "Yellow (White)" if chess_board.turn == chess.WHITE else "Red (Black)"
            cv2.putText(undistorted_frame, f"Turn: {display_turn_color}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)


        cv2.imshow('Original Feed (Undistorted)', undistorted_frame)
        cv2.imshow('Flattened Chessboard View', perspective_display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            if found and all_squares_detected_count == 64:
                # Only record pre-move state if it's the human player's turn
                if engine_ready and chess_board.turn != player_to_move_color:
                    display_current_turn = "White" if chess_board.turn == chess.WHITE else "Black"
                    print(f"It's not your turn! It's {display_current_turn}'s turn. Please wait.")
                    if speech_engine:
                        speech_engine.say(f"It's not your turn. It's {display_current_turn}'s turn.")
                        speech_engine.runAndWait()
                    continue

                pre_move_board_state = current_board_state_temp.copy()
                detection_mode = 'waiting_for_move_complete'
                print("\n--- 's' pressed: Pre-move board state recorded. Make your move now. ---")
                if speech_engine:
                    speech_engine.say("Pre move state recorded. Please make your move now.")
                    speech_engine.runAndWait()
                pending_move_data = {'move': None, 'count': 0} # Clear any pending
            else:
                print("Cannot record pre-move state: Chessboard not fully detected and stable.")
                if speech_engine:
                    speech_engine.say("Cannot record pre move state. Chessboard not stable.")
                    speech_engine.runAndWait()
        elif key == ord('m'):
            if detection_mode == 'waiting_for_move_complete':
                if found and all_squares_detected_count == 64:
                    post_move_board_state = current_board_state_temp.copy()

                    if pre_move_board_state is not None and post_move_board_state is not None:
                        moved_from_squares = []
                        moved_to_squares = []
                        captured_squares = []

                        for r_move in range(8):
                            for c_move in range(8):
                                pre_type = pre_move_board_state[r_move, c_move]
                                post_type = post_move_board_state[r_move, c_move]

                                if pre_type != post_type:
                                    if pre_type != EMPTY_SQUARE and post_type == EMPTY_SQUARE:
                                        moved_from_squares.append((r_move, c_move, pre_type))
                                    elif pre_type == EMPTY_SQUARE and post_type != EMPTY_SQUARE:
                                        moved_to_squares.append((r_move, c_move, post_type))
                                    elif pre_type != EMPTY_SQUARE and post_type != EMPTY_SQUARE:
                                        moved_to_squares.append((r_move, c_move, post_type))
                                        captured_squares.append((r_move, c_move, pre_type))

                        detected_move_uci = None
                        move_description = "No clear move detected."

                        # --- Debug prints before processing the move ---
                        print(f"\n--- DEBUG: Engine State Before Human Move ---")
                        print("Engine's Current FEN:", chess_board.fen())
                        print("Engine's Current Turn:", "White" if chess_board.turn == chess.WHITE else "Black")
                        print("Human Player Controls:", "Yellow (White)" if player_to_move_color == chess.WHITE else "Red (Black)")


                        # Only proceed with engine if it's the human player's turn
                        if engine_ready and chess_board.turn != player_to_move_color:
                            display_current_turn = "White" if chess_board.turn == chess.WHITE else "Black"
                            print(f"It's not your turn! It's {display_current_turn}'s turn. Please wait.")
                            if speech_engine:
                                speech_engine.say(f"It's not your turn. It's {display_current_turn}'s turn.")
                                speech_engine.runAndWait()
                        else:
                            # Try to find a single, unambiguous move
                            if len(moved_from_squares) == 1 and len(moved_to_squares) == 1:
                                from_square_coords, from_piece_type = moved_from_squares[0][:2], moved_from_squares[0][2]
                                to_square_coords, to_piece_type = moved_to_squares[0][:2], moved_to_squares[0][2]

                                # --- **CRITICAL CHANGE HERE: Map physical piece color to chess.COLOR** ---
                                is_moving_piece_white_in_chess = (from_piece_type == PLAYER_2_PIECE) # Yellow is White
                                is_moving_piece_black_in_chess = (from_piece_type == PLAYER_1_PIECE) # Red is Black

                                if (is_moving_piece_white_in_chess and chess_board.turn == chess.WHITE) or \
                                   (is_moving_piece_black_in_chess and chess_board.turn == chess.BLACK):
                                    
                                    from_notation = get_chessboard_notation(from_square_coords[0], from_square_coords[1])
                                    to_notation = get_chessboard_notation(to_square_coords[0], to_square_coords[1])
                                    
                                    potential_uci = from_notation.lower() + to_notation.lower()

                                    # Attempt to handle promotions (basic: assumes queen promotion if applicable)
                                    # Yellow (White) pawn reaches 8th rank (row 0)
                                    if is_moving_piece_white_in_chess and to_square_coords[0] == 0 and chess_board.piece_type_at(chess.square(from_square_coords[1], 7-from_square_coords[0])) == chess.PAWN:
                                        detected_move_uci = potential_uci + 'q'
                                        move_description = f"Detected promotion: {from_notation} to {to_notation} Queen"
                                    # Red (Black) pawn reaches 1st rank (row 7)
                                    elif is_moving_piece_black_in_chess and to_square_coords[0] == 7 and chess_board.piece_type_at(chess.square(from_square_coords[1], 7-from_square_coords[0])) == chess.PAWN:
                                        detected_move_uci = potential_uci + 'q'
                                        move_description = f"Detected promotion: {from_notation} to {to_notation} Queen"
                                    else:
                                        detected_move_uci = potential_uci
                                        move_description = f"Move: {from_notation} to {to_notation}"

                                    moving_color_physical = 'Yellow' if is_moving_piece_white_in_chess else 'Red'
                                    move_description = f"{moving_color_physical} ({'White' if chess_board.turn == chess.WHITE else 'Black'}) {move_description}"

                                else:
                                    move_description = f"Detected move by wrong physical color for current chess turn. Expected {'Yellow (White)' if chess_board.turn == chess.WHITE else 'Red (Black)'}."
                            
                            elif len(moved_from_squares) == 0 and len(moved_to_squares) == 0 and len(captured_squares) == 0:
                                move_description = "No changes detected between 's' and 'm' states."
                            else:
                                move_description = "Complex/Ambiguous change detected. (Check Console)"
                                if moved_from_squares: print(f"  Pieces left: {[get_chessboard_notation(r,c) for r,c,_ in moved_from_squares]}")
                                if moved_to_squares: print(f"  Pieces arrived: {[get_chessboard_notation(r,c) for r,c,_ in moved_to_squares]}")
                                if captured_squares: print(f"  Pieces captured/replaced on: {[get_chessboard_notation(r,c) for r,c,_ in captured_squares]}")
                                # Attempt to infer move if it's a known complex move type (e.g. castling, but this requires more specific logic)
                                # For now, for simplicity, we'll try the single move case if there's only one source and one destination.
                                if len(moved_from_squares) == 1 and len(moved_to_squares) == 1:
                                    from_sq_str = get_chessboard_notation(moved_from_squares[0][0], moved_from_squares[0][1]).lower()
                                    to_sq_str = get_chessboard_notation(moved_to_squares[0][0], moved_to_squares[0][1]).lower()
                                    potential_uci = from_sq_str + to_sq_str
                                    # Check for promotion if applicable (handled above, but keep similar fallback)
                                    if chess.Move.from_uci(potential_uci) in chess_board.legal_moves:
                                        detected_move_uci = potential_uci
                                        move_description = f"Detected complex move (as simple): {potential_uci}"
                                    else:
                                        move_description += f" - Could not infer legal move from {potential_uci}"


                            print(f"\n--- {move_description} ---")
                            if speech_engine:
                                # Replace '->' with 'to' for better pronunciation
                                speech_engine.say(move_description.replace("->", "to")) 
                                speech_engine.runAndWait()

                            print("\n--- Current Board State (0:Empty, 1:Red, 2:Yellow) ---")
                            for row_data in post_move_board_state:
                                print("".join(map(str, row_data)))
                            print("---------------------------\n")

                            # --- Process Human Move with Chess Engine ---
                            if engine_ready and detected_move_uci:
                                try:
                                    human_move = chess.Move.from_uci(detected_move_uci)
                                    
                                    print(f"DEBUG: Checking if {human_move.uci()} is legal on engine board...")
                                    if human_move in chess_board.legal_moves:
                                        chess_board.push(human_move)
                                        print(f"Human move '{human_move.uci()}' successfully applied to engine board.")
                                        print("Board FEN after human move: ", chess_board.fen())

                                        if chess_board.is_game_over():
                                            print("\n--- GAME OVER ---")
                                            print(f"Result: {chess_board.result()}")
                                            if speech_engine:
                                                speech_engine.say(f"Game over. {chess_board.result()}")
                                                speech_engine.runAndWait()
                                        else:
                                            print(f"\n--- Engine ({'White' if chess_board.turn == chess.WHITE else 'Black'}) is thinking... ---")
                                            # AWAIT the engine.play() call
                                            engine_result = await chess_engine.play(chess_board, chess.engine.Limit(time=0.5))
                                            engine_move = engine_result.move
                                            chess_board.push(engine_move)
                                            # Corrected print line for square names
                                            engine_move_text = f"Engine move: {chess.square_name(engine_move.from_square)} to {chess.square_name(engine_move.to_square)}"
                                            print(engine_move_text)
                                            print("Board FEN after engine move: ", chess_board.fen())
                                            print("Please make the engine's move on the physical board.")

                                            # Speak the engine's move
                                            if speech_engine:
                                                speech_engine.say(engine_move_text)
                                                speech_engine.runAndWait()

                                            if chess_board.is_game_over():
                                                print("\n--- GAME OVER ---")
                                                print(f"Result: {chess_board.result()}")
                                                if speech_engine:
                                                    speech_engine.say(f"Game over. {chess_board.result()}")
                                                    speech_engine.runAndWait()

                                    else:
                                        print(f"ILLEGAL MOVE: '{detected_move_uci}' is not a legal move according to the engine's current state.")
                                        print("Legal moves are: ", [m.uci() for m in chess_board.legal_moves])
                                        if speech_engine:
                                            speech_engine.say(f"Illegal move: {detected_move_uci}. Please try again.")
                                            speech_engine.runAndWait()
                                except ValueError as ve:
                                    print(f"Error parsing detected move '{detected_move_uci}': {ve}. This might be an invalid UCI string or incorrect parsing for promotion.")
                                    if speech_engine:
                                        speech_engine.say(f"Error parsing move. {detected_move_uci}.")
                                        speech_engine.runAndWait()
                                except Exception as ex:
                                    print(f"An unexpected error occurred during engine interaction: {ex}")
                                    if speech_engine:
                                        speech_engine.say("An error occurred with the chess engine.")
                                        speech_engine.runAndWait()
                            elif not engine_ready:
                                print("Chess engine not initialized. Cannot process moves.")
                                if speech_engine:
                                    speech_engine.say("Chess engine not ready.")
                                    speech_engine.runAndWait()


                    else:
                        print("Error: Pre-move board state not recorded or board became unstable.")
                        if speech_engine:
                            speech_engine.say("Error. Pre move state not recorded or board is unstable.")
                            speech_engine.runAndWait()

                    # Reset for next move detection cycle
                    detection_mode = 'waiting_for_start'
                    pre_move_board_state = None
                    post_move_board_state = None
                    pending_move_data = {'move': None, 'count': 0}
                else:
                    print("Cannot confirm move: Chessboard not fully detected and stable after move. Please ensure board is clear and try 'm' again, or 's' to restart.")
                    if speech_engine:
                        speech_engine.say("Cannot confirm move. Chessboard not stable. Please try again.")
                        speech_engine.runAndWait()
            else:
                print("Press 's' first to start a move, then 'm' to confirm.")
                if speech_engine:
                    speech_engine.say("Please press s to start your move.")
                    speech_engine.runAndWait()


        # Give a small pause to the event loop so other async tasks can run
        await asyncio.sleep(0.01) # Small non-blocking sleep

    cap.release()
    cv2.destroyAllWindows()
    if engine_ready:
        await chess_engine.quit() # Use await for quitting
    # Stop speech engine when quitting
    if speech_engine:
        speech_engine.stop()
    print("Camera released and windows closed. Chess engine quit.")
    print("Speech engine stopped.") # Added this for confirmation

if __name__ == "__main__":
    asyncio.run(capture_live_video_feed())
