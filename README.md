# ♟️ ChessEngine — A Computer-Vision Powered Physical Chess Board

Play a real, physical game of chess against **Stockfish** using nothing but a webcam, a printed chessboard, and two colors of paint. No sensors under the squares, no smart pieces, no RFID — just OpenCV watching the board and figuring out what happened.

> Move a real piece → the camera sees the change → the engine validates and replies → your speakers tell you what to play next.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Features](#features)
- [Hardware You'll Need](#hardware-youll-need)
- [Getting Started](#getting-started)
- [Controls](#controls)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Companion Hardware: Keypad + LCD Input](#companion-hardware-keypad--lcd-input)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

ChessEngine turns an ordinary chessboard into an interactive, engine-backed opponent. A single webcam mounted above the board tracks the position after every move using classic computer vision — camera calibration, perspective correction, and HSV color thresholding — with no machine learning model and no special hardware. Detected moves are validated against the rules of chess and sent to **Stockfish**, whose replies are spoken aloud so you never have to look away from the board.

It's a compact demonstration of computer vision, geometric transforms, and real-time systems integration all in one physical object.

## How It Works

The pipeline runs in five stages, all inside `chesscam.py`:

1. **Camera Calibration** — `calibration.py` captures ~10–15 views of a printed chessboard pattern and uses `cv2.calibrateCamera` to compute a camera matrix and distortion coefficients, correcting for lens distortion.
2. **Board Detection & Perspective Warp** — Every frame, `cv2.findChessboardCorners` locates the board's inner corners. The four outer corners are extrapolated and used with `cv2.getPerspectiveTransform` to warp the board into a flat, top-down 640×640 image, split into a clean 8×8 grid.
3. **Piece Detection (HSV Color Segmentation)** — Each of the 64 warped squares is converted to HSV and thresholded against tuned red and yellow ranges. Pixel counts above a configurable threshold mark a square as occupied by that color.
4. **Move Inference** — Press `s` to snapshot the board before your move, make the move, then press `m`. The program diffs the "before" and "after" 8×8 states: a square that lost a piece is the origin, a square that gained one is the destination. Pawn promotions to the back rank are auto-detected and default to queen promotion.
5. **Engine Play & Voice Feedback** — The inferred move is validated with `python-chess`. If legal, it's pushed to the board and Stockfish (via `chess.engine.popen_uci`) computes a reply, which is printed, spoken with `pyttsx3`, and waits for you to physically make it on the board.

## Features

- 🎥 Pure computer-vision move detection — no embedded sensors in the board or pieces
- ♞ Full legal-move validation via `python-chess`, including auto-queen promotion
- 🤖 Real engine opponent (Stockfish, UCI protocol, configurable think time)
- 🔊 Spoken turn announcements, move confirmations, and error/illegal-move feedback
- 📐 One-time camera calibration workflow with saved `.npy` calibration files
- 🎨 Interactive HSV color-picker tool for tuning piece detection to your lighting and paint

## Hardware You'll Need

| Item | Notes |
|---|---|
| Standard 8×8 chessboard | Any size; the checkerboard corner pattern is used for calibration and tracking |
| Webcam | Mounted directly above the board, full board in frame |
| Chess pieces | White pieces painted **yellow**, black pieces painted **red** (or use pieces that are naturally these colors) — the detector treats physical yellow as chess White and physical red as chess Black |
| A computer running the Python scripts | Windows, macOS, or Linux |
| *(Optional)* Arduino + 4×4 keypad + 16×2 I2C LCD | Manual move-entry companion device — see [Companion Hardware](#companion-hardware-keypad--lcd-input) |

## Getting Started

### Prerequisites

```bash
pip install opencv-python numpy python-chess pyttsx3
```

You'll also need a **Stockfish** binary for your OS — download it from [stockfishchess.org](https://stockfishchess.org/download/) and note its path.

### Installation

```bash
git clone https://github.com/mkartsiotis/chessengine.git
cd chessengine
```

### Setup

1. **Calibrate your camera:**
   ```bash
   python calibration.py
   ```
   Hold the chessboard pattern in front of the camera at different angles/distances. Press `s` to save each good view (aim for 10–15), then `c` to run calibration once you have enough. This writes `camera_matrix.npy` and `dist_coeffs.npy`.

2. **(Optional) Tune piece-color detection:**
   ```bash
   python hsvscan.py
   ```
   Point it at a saved photo of your board (edit the filename in the script) and click on pieces to read off their HSV values. Use these to adjust `lower_red1/2`, `upper_red1/2`, `lower_yellow`, and `upper_yellow` in `chesscam.py`.

3. **Point the script at your Stockfish binary:**
   Open `chesscam.py` and set:
   ```python
   STOCKFISH_PATH = r"/path/to/your/stockfish"
   ```

4. **Play:**
   ```bash
   python chesscam.py
   ```

## Controls

| Key | Action |
|---|---|
| `s` | Record the board state **before** you move (start of turn) |
| `m` | Record the board state **after** you move and infer the move |
| `q` | Quit |

## Project Structure

| File | Purpose |
|---|---|
| `chesscam.py` | Main application: detection, move inference, engine integration, TTS |
| `calibration.py` | Camera calibration tool — generates the `.npy` files below |
| `hsvscan.py` | Standalone HSV color-picker utility for threshold tuning |
| `camera_matrix.npy`, `dist_coeffs.npy` | Saved calibration output (camera-specific — regenerate for your own setup) |
| `dwsasa.py`, `frrsxgfgh.py`, `sdfghjk.py`, `skdjd.py`, `scanscanscan.py` | Development/exploration scripts from earlier iterations |

> **Suggested cleanup:** the exploratory scripts have placeholder names left over from development. Moving them into an `experiments/` folder (or removing the ones that are fully superseded by `chesscam.py`) would make the repo read a lot cleaner for anyone browsing it.

## Configuration

Key tunables live at the top of `chesscam.py`:

```python
lower_red1 / upper_red1   # first HSV range for red (wraps around hue 0)
lower_red2 / upper_red2   # second HSV range for red (wraps around hue 180)
lower_yellow / upper_yellow

red_pixel_threshold = 15      # min red pixels in a square to count as occupied
yellow_pixel_threshold = 90   # min yellow pixels in a square to count as occupied

player_to_move_color = chess.WHITE   # which physical color the human plays
```

Lighting changes between rooms/times of day are the most common cause of misdetection — re-run `hsvscan.py` if pieces stop being recognized reliably.

## Companion Hardware: Keypad + LCD Input

Alongside the camera-vision system, there's an Arduino-based manual input device: a 4×4 matrix keypad plus a 16×2 I2C LCD that lets a player key in moves by hand and displays the engine's reply — useful as a fallback if the camera can't see the board, or for testing move logic without a camera at all. **This isn't wired into `chesscam.py` yet** (there's no serial code there currently) — think of it as a self-contained module worth adding to the repo under something like `hardware/keypad-lcd-interface/`.

**Wiring**

| Component | Pins |
|---|---|
| Keypad rows | D9, D8, D7, D6 |
| Keypad columns | D5, D4, D3, D2 |
| LCD | I2C, address `0x27`, 16×2 |

**Libraries:** `Keypad.h`, `Wire.h`, `LiquidCrystal_I2C.h`

**Serial protocol (as currently implemented in the sketch):**

| Host sends | Arduino does |
|---|---|
| `"1"` | Shows "It is your turn," waits for a from/to square entered on the keypad, sends back `"<from>,<to>"` (e.g. `e2,h8`) |
| `"CPU"` followed by a move string | Shows `"COMPUTER MOVE: <move>"`, waits for any keypress, replies `"VER"` |
| Anything else | Stops the move-entry loop |

**Keypad key → square mapping**

*First press (file, a–h):*

| Key | File | Key | File |
|---|---|---|---|
| `1` | a | `4` | e |
| `A` | b | `B` | f |
| `2` | c | `5` | g |
| `3` | d | `6` | h |

*Second press (rank, 1–8):*

| Key | Rank | Key | Rank |
|---|---|---|---|
| `1` | 1 | `4` | 5 |
| `A` | 2 | `5` | 7 |
| `2` | 3 | `6` | 8 |
| `3` | 4 | `B` | 6 |

Pressing `9` on the second entry confirms and sends the move. Any unrecognized key at either stage triggers an "UNDO" and resets the current entry.

## Known Limitations

- HSV thresholds are sensitive to ambient lighting — a setup calibrated in daylight may misread pieces under indoor lighting.
- Move inference currently handles single, unambiguous from/to square changes cleanly; special moves like castling and en passant fall into the "complex/ambiguous" branch and print to console rather than being auto-detected.
- `camera_matrix.npy` / `dist_coeffs.npy` are specific to the camera they were generated on — recalibrate if you change webcams.
- The Stockfish path is currently a hardcoded absolute path rather than a config value or CLI argument.

## Roadmap

- [ ] Bridge the keypad/LCD device into `chesscam.py` over serial as an alternate input mode
- [ ] Robotic XY gantry to physically move pieces after the engine's reply, closing the loop end-to-end
- [ ] Native handling of castling, en passant, and under-promotion in move inference
- [ ] Config file / CLI args for Stockfish path, thresholds, and player color (remove hardcoded paths)
- [ ] Lightweight GUI overlay showing the live board state and move history

## Contributing

Issues and pull requests are welcome — especially around lighting robustness, special-move detection, and the hardware integration pieces above.

## License

No license file is currently included. Adding one (MIT is a common, permissive choice for a portfolio project like this) makes it clear what others are allowed to do with the code.

## Acknowledgments

- [OpenCV](https://opencv.org/) — board detection, calibration, perspective transform
- [python-chess](https://python-chess.readthedocs.io/) — move validation and board state
- [Stockfish](https://stockfishchess.org/) — the engine opponent
- [pyttsx3](https://pyttsx3.readthedocs.io/) — offline text-to-speech
