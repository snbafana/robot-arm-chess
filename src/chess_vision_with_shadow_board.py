"""
chess_vision_squares.py
---------------------
Capture and analyze chess moves using computer vision.
Simplified version focusing on board configuration and move detection.
"""

import cv2
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from shadow_board import sync_shadow_board, save_fen_history, export_game_positions
from stockfish_api import analyze_position_with_stockfish, print_stockfish_analysis

# Setup
CAPTURES_DIR = Path("captures")
CAPTURES_DIR.mkdir(exist_ok=True)

# Board dimensions
BOARD_SIZE = (8, 8)  # 8x8 chessboard
SQUARE_SIZE = 100    # Size of each square in pixels

# Piece symbols for visualization
PIECE_SYMBOLS = {
    "white": {
        "pawn": "♙",
        "knight": "♘",
        "bishop": "♗",
        "rook": "♖",
        "queen": "♕",
        "king": "♔"
    },
    "black": {
        "pawn": "♟",
        "knight": "♞",
        "bishop": "♝",
        "rook": "♜",
        "queen": "♛",
        "king": "♚"
    }
}

# Define the initial chessboard state
INITIAL_BOARD = {
    "a1": {"type": "rook", "color": "white"},
    "b1": {"type": "knight", "color": "white"},
    "c1": {"type": "bishop", "color": "white"},
    "d1": {"type": "queen", "color": "white"},
    "e1": {"type": "king", "color": "white"},
    "f1": {"type": "bishop", "color": "white"},
    "g1": {"type": "knight", "color": "white"},
    "h1": {"type": "rook", "color": "white"},
    "a2": {"type": "pawn", "color": "white"},
    "b2": {"type": "pawn", "color": "white"},
    "c2": {"type": "pawn", "color": "white"},
    "d2": {"type": "pawn", "color": "white"},
    "e2": {"type": "pawn", "color": "white"},
    "f2": {"type": "pawn", "color": "white"},
    "g2": {"type": "pawn", "color": "white"},
    "h2": {"type": "pawn", "color": "white"},
    "a8": {"type": "rook", "color": "black"},
    "b8": {"type": "knight", "color": "black"},
    "c8": {"type": "bishop", "color": "black"},
    "d8": {"type": "queen", "color": "black"},
    "e8": {"type": "king", "color": "black"},
    "f8": {"type": "bishop", "color": "black"},
    "g8": {"type": "knight", "color": "black"},
    "h8": {"type": "rook", "color": "black"},
    "a7": {"type": "pawn", "color": "black"},
    "b7": {"type": "pawn", "color": "black"},
    "c7": {"type": "pawn", "color": "black"},
    "d7": {"type": "pawn", "color": "black"},
    "e7": {"type": "pawn", "color": "black"},
    "f7": {"type": "pawn", "color": "black"},
    "g7": {"type": "pawn", "color": "black"},
    "h7": {"type": "pawn", "color": "black"}
}

def load_config() -> Dict:
    """Load configuration from config file."""
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {
        "camera": {
            "index": 1,
            "is_calibrated": False,
            "settings": {
                "exposure": -4,
                "brightness": 30,
                "contrast": 50,
                "saturation": 50,
                "gain": 30
            }
        },
        "calibration": {
            "square_size": SQUARE_SIZE,
            "matrix": None,
            "last_calibration": None
        },
        "board": {
            "position": None,
            "size": None,
            "square_size": SQUARE_SIZE
        }
    }

def save_config(config: Dict):
    """Save configuration to file."""
    with open("config.json", 'w') as f:
        json.dump(config, f, indent=2)

def adjust_camera_settings(camera_index: int, config: Dict) -> Dict:
    """Adjust camera settings interactively."""
    print("\n=== Camera Settings Adjustment ===")
    print("Use the following keys to adjust settings:")
    print("b: brightness (+/-)")
    print("c: contrast (+/-)")
    print("e: exposure (+/-)")
    print("s: saturation (+/-)")
    print("g: gain (+/-)")
    print("r: reset to defaults")
    print("q: quit and save")
    
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {camera_index} not accessible.")
    
    # Set initial settings
    settings = config["camera"]["settings"]
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, settings["exposure"])
    cap.set(cv2.CAP_PROP_BRIGHTNESS, settings["brightness"])
    cap.set(cv2.CAP_PROP_CONTRAST, settings["contrast"])
    cap.set(cv2.CAP_PROP_SATURATION, settings["saturation"])
    cap.set(cv2.CAP_PROP_GAIN, settings["gain"])
    
    step = 5  # Adjustment step size
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Display current settings
        cv2.putText(frame, f"Brightness: {settings['brightness']}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Contrast: {settings['contrast']}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Exposure: {settings['exposure']}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Saturation: {settings['saturation']}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Gain: {settings['gain']}", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Camera Settings", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('b'):  # Brightness
            settings["brightness"] = min(100, max(0, settings["brightness"] + step))
            cap.set(cv2.CAP_PROP_BRIGHTNESS, settings["brightness"])
        elif key == ord('c'):  # Contrast
            settings["contrast"] = min(100, max(0, settings["contrast"] + step))
            cap.set(cv2.CAP_PROP_CONTRAST, settings["contrast"])
        elif key == ord('e'):  # Exposure
            settings["exposure"] = min(0, max(-10, settings["exposure"] - step))
            cap.set(cv2.CAP_PROP_EXPOSURE, settings["exposure"])
        elif key == ord('s'):  # Saturation
            settings["saturation"] = min(100, max(0, settings["saturation"] + step))
            cap.set(cv2.CAP_PROP_SATURATION, settings["saturation"])
        elif key == ord('g'):  # Gain
            settings["gain"] = min(100, max(0, settings["gain"] + step))
            cap.set(cv2.CAP_PROP_GAIN, settings["gain"])
        elif key == ord('r'):  # Reset
            settings = {
                "exposure": -4,
                "brightness": 30,
                "contrast": 50,
                "saturation": 50,
                "gain": 30
            }
            cap.set(cv2.CAP_PROP_EXPOSURE, settings["exposure"])
            cap.set(cv2.CAP_PROP_BRIGHTNESS, settings["brightness"])
            cap.set(cv2.CAP_PROP_CONTRAST, settings["contrast"])
            cap.set(cv2.CAP_PROP_SATURATION, settings["saturation"])
            cap.set(cv2.CAP_PROP_GAIN, settings["gain"])
        elif key == ord('q'):  # Quit
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Update config with new settings
    config["camera"]["settings"] = settings
    save_config(config)
    return config

def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order corners in [top-left, top-right, bottom-right, bottom-left] order."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def create_chess_grid_overlay(image: np.ndarray, highlighted_squares: List[Tuple[int, int]] = None) -> np.ndarray:
    """Create a chess grid overlay on the image.
    
    Args:
        image: Input image to overlay grid on
        highlighted_squares: Optional list of (rank, file) tuples to highlight
    
    Returns:
        Image with grid overlay
    """
    height, width = image.shape[:2]
    
    # Create visualization
    if len(image.shape) == 2:  # If grayscale
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()
    
    # Draw grid lines
    for i in range(9):  # 9 lines for 8 squares
        # Vertical lines
        x = (i * width) // 8
        cv2.line(vis_image, (x, 0), (x, height), (0, 255, 0), 2)
        
        # Horizontal lines
        y = (i * height) // 8
        cv2.line(vis_image, (0, y), (width, y), (0, 255, 0), 2)
    
    # Add square labels (A1 through H8)
    for i in range(8):  # files (A-H, vertical)
        for j in range(8):  # ranks (1-8, horizontal)
            # Files go top to bottom (A-H), ranks go left to right (1-8)
            file_letter = chr(ord('A') + i)  # A-H
            rank_number = str(j + 1)  # 1-8
            square_name = f"{file_letter}{rank_number}"
            
            # Position labels in top-left of each square
            x = (j * width) // 8 + 5  # j for rank (horizontal)
            y = (i * height) // 8 + 20  # i for file (vertical)
            
            # Draw label
            cv2.putText(vis_image, square_name, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Highlight specified squares if any
    if highlighted_squares:
        square_height = height // 8
        square_width = width // 8
        
        for rank, file in highlighted_squares:
            # Convert coordinates to match new orientation
            x_start = (rank - 1) * square_width  # rank is horizontal (1-8)
            y_start = (ord(file) - ord('A')) * square_height  # file is vertical (A-H)
            
            # Draw filled rectangle with semi-transparency
            overlay = vis_image.copy()
            cv2.rectangle(overlay, 
                         (x_start, y_start),
                         (x_start + square_width, y_start + square_height),
                         (0, 255, 0), -1)  # -1 for filled rectangle
            # Apply transparency
            cv2.addWeighted(overlay, 0.3, vis_image, 0.7, 0, vis_image)
            # Draw border
            cv2.rectangle(vis_image, 
                         (x_start, y_start),
                         (x_start + square_width, y_start + square_height),
                         (0, 255, 0), 2)
    
    return vis_image

def configure_board(camera_index: int) -> Tuple[bool, Dict]:
    """Configure the chessboard in two steps: corner selection and board selection."""
    config = load_config()
    
    # Step 1: Camera Settings Adjustment
    print("\n=== Camera Settings Adjustment ===")
    config = adjust_camera_settings(camera_index, config)
    
    # Step 2: Corner Selection
    print("\n=== Corner Selection ===")
    print("1. Click on the four corners of the chessboard in this order:")
    print("   - Top-left corner")
    print("   - Top-right corner")
    print("   - Bottom-right corner")
    print("   - Bottom-left corner")
    print("2. Press 'r' to reset if you make a mistake")
    print("3. Press 'c' to confirm when all corners are selected")
    print("4. Press 'q' to quit\n")
    
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {camera_index} not accessible.")
    
    # Set camera properties
    settings = config["camera"]["settings"]
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, settings["exposure"])
    cap.set(cv2.CAP_PROP_BRIGHTNESS, settings["brightness"])
    cap.set(cv2.CAP_PROP_CONTRAST, settings["contrast"])
    cap.set(cv2.CAP_PROP_SATURATION, settings["saturation"])
    cap.set(cv2.CAP_PROP_GAIN, settings["gain"])
    
    corners = []
    image_copy = None
    
    def click_event(event, x, y, flags, param):
        nonlocal corners, image_copy
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append((x, y))
            cv2.circle(image_copy, (x, y), 5, (0, 255, 0), -1)
            if len(corners) > 1:
                cv2.line(image_copy, corners[-2], corners[-1], (0, 255, 0), 2)
            if len(corners) == 4:
                cv2.line(image_copy, corners[-1], corners[0], (0, 255, 0), 2)
    
    cv2.namedWindow("Corner Selection")
    cv2.setMouseCallback("Corner Selection", click_event)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        image_copy = frame.copy()
        
        # Draw current corners
        for i, (x, y) in enumerate(corners):
            cv2.circle(image_copy, (x, y), 5, (0, 255, 0), -1)
            if i > 0:
                cv2.line(image_copy, corners[i-1], (x, y), (0, 255, 0), 2)
        if len(corners) == 4:
            cv2.line(image_copy, corners[-1], corners[0], (0, 255, 0), 2)
        
        cv2.imshow("Corner Selection", image_copy)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            corners = []
        elif key == ord('c') and len(corners) == 4:
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return False, config
    
    cv2.destroyAllWindows()
    
    # Step 3: Board Selection
    print("\n=== Board Selection ===")
    print("1. Click and drag to select the entire chessboard")
    print("2. Press 's' to save the selection")
    print("3. Press 'r' to reset")
    print("4. Press 'q' to quit")
    
    drawing = False
    ix, iy = -1, -1
    fx, fy = -1, -1
    board_width = 0
    board_height = 0
    
    def draw_rectangle(event, x, y, flags, param):
        nonlocal ix, iy, fx, fy, drawing, board_width, board_height
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                fx, fy = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            fx, fy = x, y
            board_width = abs(fx - ix)
            board_height = abs(fy - iy)
    
    cv2.namedWindow("Board Selection")
    cv2.setMouseCallback("Board Selection", draw_rectangle)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Apply perspective transform using corners
        if len(corners) == 4:
            corners_array = np.array(corners, dtype="float32")
            ordered_corners = order_corners(corners_array)
            board_width = 8 * SQUARE_SIZE
            board_height = 8 * SQUARE_SIZE
            dst_points = np.array([
                [0, 0],
                [board_width - 1, 0],
                [board_width - 1, board_height - 1],
                [0, board_height - 1]
            ], dtype="float32")
            matrix = cv2.getPerspectiveTransform(ordered_corners, dst_points)
            frame = cv2.warpPerspective(frame, matrix, (board_width, board_height))
        
        # Draw selection rectangle
        if ix != -1 and iy != -1:
            if drawing:
                cv2.rectangle(frame, (ix, iy), (fx, fy), (0, 255, 0), 2)
            elif fx != -1 and fy != -1 and board_width > 0 and board_height > 0:
                cv2.rectangle(frame, (ix, iy), (fx, fy), (0, 255, 0), 2)
        
        cv2.imshow("Board Selection", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            if board_width > 0 and board_height > 0:
                break
            else:
                print("Please select the chessboard first")
        elif key == ord('r'):
            ix, iy = -1, -1
            fx, fy = -1, -1
            board_width = 0
            board_height = 0
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return False, config
    
    # After board selection loop ends and before Step 4
    # Save the selected region
    print("\n=== Saving Selected Region ===")
    
    # Ensure coordinates are in the correct order (top-left to bottom-right)
    x_start = min(ix, fx)
    x_end = max(ix, fx)
    y_start = min(iy, fy)
    y_end = max(iy, fy)
    
    # Ensure coordinates are within frame bounds
    height, width = frame.shape[:2]
    x_start = max(0, x_start)
    x_end = min(width, x_end)
    y_start = max(0, y_start)
    y_end = min(height, y_end)
    
    # Check if we have valid coordinates
    if x_start >= x_end or y_start >= y_end:
        print("Invalid selection coordinates. Please try again.")
        return False, config
    
    # Crop the selected region
    selected_region = frame[y_start:y_end, x_start:x_end]
    
    # Verify the selected region is not empty
    if selected_region.size == 0:
        print("Selected region is empty. Please try again.")
        return False, config
    
    selected_path = str(CAPTURES_DIR / "selected_region.jpg")
    cv2.imwrite(selected_path, selected_region)
    print(f"Selected region saved to: {selected_path}")
    
    # Step 4: Show chess notation grid
    print("\n=== Chess Notation Grid ===")
    print("Showing chess notation grid. Press any key to continue...")
    
    # Create grid visualization
    grid_frame = create_chess_grid_overlay(selected_region)
    
    # Show the grid
    cv2.imshow("Chess Notation Grid", grid_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Print square dimensions
    print(f"\nSquare dimensions: {board_width // 8}x{board_height // 8} pixels")
    print("Chess notation grid shown. Configuration complete.")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Update config with new settings
    config["camera"]["is_calibrated"] = True
    config["calibration"]["matrix"] = matrix.tolist()
    config["calibration"]["last_calibration"] = datetime.now().isoformat()
    config["board"]["position"] = (ix, iy)
    config["board"]["size"] = (board_width, board_height)
    config["board"]["square_size"] = min(board_width, board_height) // 8
    
    save_config(config)
    return True, config

def capture_board(camera_index: int, config: Dict) -> np.ndarray:
    """Capture and process chess board image."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {camera_index} not accessible.")
    
    # Set camera properties
    settings = config["camera"]["settings"]
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, settings["exposure"])
    cap.set(cv2.CAP_PROP_BRIGHTNESS, settings["brightness"])
    cap.set(cv2.CAP_PROP_CONTRAST, settings["contrast"])
    cap.set(cv2.CAP_PROP_SATURATION, settings["saturation"])
    cap.set(cv2.CAP_PROP_GAIN, settings["gain"])
    
    for _ in range(5): cap.read()  # Let exposure settle
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise RuntimeError("Failed to capture frame.")
    
    frame = cv2.convertScaleAbs(frame, alpha=0.9, beta=-5)
    
    # Apply perspective transform if calibrated
    if config["camera"]["is_calibrated"] and config["calibration"]["matrix"] is not None:
        matrix = np.array(config["calibration"]["matrix"])
        square_size = config["calibration"]["square_size"]
        board_width = 8 * square_size
        board_height = 8 * square_size
        
        # Apply perspective transform
        frame = cv2.warpPerspective(frame, matrix, (board_width, board_height))
        
        # Get board position and size from config
        ix, iy = config["board"]["position"]
        board_width, board_height = config["board"]["size"]
        
        # Crop to the selected region
        frame = frame[iy:iy+board_height, ix:ix+board_width]
    
    return frame

def save_image(image: np.ndarray, path: str) -> str:
    """Save image with error handling."""
    if not str(path).lower().endswith('.jpg'):
        path = str(Path(path).with_suffix('.jpg'))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(path, image)
    if not success:
        raise RuntimeError(f"Failed to save image to {path}")
    return path

def analyze_move_from_diff(diff_image: np.ndarray, initial_image: np.ndarray, current_image: np.ndarray) -> Tuple[str, str]:
    """Analyze diff image to detect piece movement.
    
    Args:
        diff_image: The difference image between initial and current
        initial_image: The initial board state image
        current_image: The current board state image
    
    Returns:
        Tuple of (from_square, to_square) in chess notation
    """
    # Get image dimensions
    height, width = diff_image.shape[:2]
    
    # Calculate square dimensions
    square_height = height // 8
    square_width = width // 8
    
    # Convert all images to grayscale if they're in color
    if len(diff_image.shape) == 3:
        diff_gray = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)
        initial_gray = cv2.cvtColor(initial_image, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    else:
        diff_gray = diff_image
        initial_gray = initial_image
        current_gray = current_image
    
    # Initialize array to store average intensity for each square
    square_intensities = np.zeros((8, 8))
    
    # Calculate average intensity for each square in diff image
    for i in range(8):  # files (vertical)
        for j in range(8):  # ranks (horizontal)
            # Get square region
            y_start = i * square_height
            y_end = (i + 1) * square_height
            x_start = j * square_width
            x_end = (j + 1) * square_width
            
            square = diff_gray[y_start:y_end, x_start:x_end]
            square_intensities[i, j] = np.mean(square)
    
    # Find the two squares with highest intensity changes
    flat_intensities = square_intensities.flatten()
    top_two_indices = np.argsort(flat_intensities)[-2:]
    
    # Get the squares' coordinates
    squares = []
    for idx in top_two_indices:
        file_idx = idx // 8  # Vertical position (A-H)
        rank = idx % 8 + 1   # Horizontal position (1-8)
        
        # Get average intensities for this square in both initial and current images
        y_start = file_idx * square_height
        y_end = (file_idx + 1) * square_height
        x_start = (rank - 1) * square_width
        x_end = rank * square_width
        
        initial_square = initial_gray[y_start:y_end, x_start:x_end]
        current_square = current_gray[y_start:y_end, x_start:x_end]
        
        initial_intensity = np.mean(initial_square)
        current_intensity = np.mean(current_square)
        
        file_letter = chr(ord('A') + file_idx)
        squares.append({
            'notation': f"{file_letter}{rank}",
            'initial_intensity': initial_intensity,
            'current_intensity': current_intensity
        })
    
    # Determine which square is the source (had piece initially, now empty)
    # and which is destination (was empty, now has piece)
    if abs(squares[0]['initial_intensity'] - squares[0]['current_intensity']) > \
       abs(squares[1]['initial_intensity'] - squares[1]['current_intensity']):
        # Square 0 had bigger change
        if squares[0]['initial_intensity'] > squares[0]['current_intensity']:
            # Square 0 got darker (piece moved away)
            from_square, to_square = squares[0]['notation'], squares[1]['notation']
        else:
            # Square 0 got lighter (piece moved here)
            from_square, to_square = squares[1]['notation'], squares[0]['notation']
    else:
        # Square 1 had bigger change
        if squares[1]['initial_intensity'] > squares[1]['current_intensity']:
            # Square 1 got darker (piece moved away)
            from_square, to_square = squares[1]['notation'], squares[0]['notation']
        else:
            # Square 1 got lighter (piece moved here)
            from_square, to_square = squares[0]['notation'], squares[1]['notation']
    
    # Create visualization with grid overlay
    vis_image = create_chess_grid_overlay(diff_gray, [(int(to_square[1]), to_square[0])])
    
    # Add arrow from source to destination
    start_x = (int(from_square[1]) - 1) * square_width + square_width // 2
    start_y = (ord(from_square[0]) - ord('A')) * square_height + square_height // 2
    end_x = (int(to_square[1]) - 1) * square_width + square_width // 2
    end_y = (ord(to_square[0]) - ord('A')) * square_height + square_height // 2
    
    # Draw arrow
    cv2.arrowedLine(vis_image, (start_x, start_y), (end_x, end_y),
                    (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.2)
    
    # Add move text
    move_text = f"Move: {from_square} -> {to_square}"
    cv2.putText(vis_image, move_text, (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Save visualization
    cv2.imwrite(str(CAPTURES_DIR / "move_detection.jpg"), vis_image)
    
    return from_square, to_square

def print_board(board_state: Dict):
    """Print the current board state in ASCII format."""
    print("\nCurrent Board State:")
    print("  a  b  c  d  e  f  g  h")
    print("  ─  ─  ─  ─  ─  ─  ─  ─")
    
    for rank in range(8, 0, -1):
        print(f"{rank}│", end="")
        for file in "abcdefgh":
            square = f"{file}{rank}"
            if square in board_state:
                piece = board_state[square]
                symbol = PIECE_SYMBOLS[piece["color"]][piece["type"]]
                print(f" {symbol} ", end="│")
            else:
                print("   ", end="│")
        print(f"{rank}")
        if rank > 1:
            print(" │", end="")
            print("─" * 3 + "┼" + "─" * 3, end="")
            for _ in range(6):
                print("┼" + "─" * 3, end="")
            print("│")
    
    print("  ─  ─  ─  ─  ─  ─  ─  ─")
    print("  a  b  c  d  e  f  g  h\n")

def update_board_state(board_state: Dict, from_square: str, to_square: str) -> Dict:
    """Update the board state based on a move."""
    new_state = board_state.copy()
    
    # Get the piece that's moving
    if from_square in new_state:
        piece = new_state[from_square]
        # Remove piece from old square
        del new_state[from_square]
        # Add piece to new square (this will automatically handle captures)
        new_state[to_square] = piece
    
    return new_state

def save_moves(session_dir: Path, moves: List[Dict]):
    """Save move history."""
    moves_file = session_dir / "moves.json"
    with open(moves_file, 'w') as f:
        json.dump(moves, f, indent=2)

def save_board_states(session_dir: Path, board_states: List[Dict]):
    """Save board states history."""
    states_file = session_dir / "board.json"
    with open(states_file, 'w') as f:
        json.dump(board_states, f, indent=2)

def main():
    """Main execution loop."""
    try:
        # Setup
        session_dir = CAPTURES_DIR / f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir.mkdir(exist_ok=True)
        print(f"\nStarting new game session in {session_dir.name}")
        
        # Camera setup - specifically use camera index 1
        cam_idx = 1
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            sys.exit(f"Camera {cam_idx} not found. Please ensure camera is connected.")
        cap.release()
        print(f"Using camera {cam_idx}")
        
        # Load or create config
        config = load_config()
        
        # Initialize game state
        board_state = INITIAL_BOARD.copy()
        move_number = 1
        is_white_move = True
        moves = []
        board_states = [{
            "timestamp": datetime.now().isoformat(),
            "move_number": 0,
            "player": "initial",
            "board": INITIAL_BOARD
        }]
        
        # Initialize shadow board and FEN history
        shadow_board = sync_shadow_board(board_state)
        fen_history = [shadow_board.fen()]
        
        # Ask if user wants to configure
        if config["camera"]["is_calibrated"]:
            print("\nCurrent configuration exists. Would you like to:")
            print("1. Use existing configuration")
            print("2. Reconfigure camera and board")
            choice = input("Enter choice (1 or 2): ").strip()
            if choice == "2":
                print("\nStarting configuration process...")
                success, config = configure_board(cam_idx)
                if not success:
                    sys.exit("Board configuration failed.")
        else:
            print("\nNo configuration exists. Starting configuration process...")
            success, config = configure_board(cam_idx)
            if not success:
                sys.exit("Board configuration failed.")
        
        # Print initial board state
        print_board(board_state)
        
        # Capture initial board state
        print("\nCapturing initial board state...")
        initial = capture_board(cam_idx, config)
        initial_path = save_image(initial, str(session_dir / "initial_board"))
        print("Initial board captured.")
        
        # Main game loop
        while True:
            print(f"\nMove {move_number} - {'White' if is_white_move else 'Black'}'s turn")
            print("Press Enter when the move is made...")
            choice = input().lower().strip()
            
            if choice == 'exit':
                print("\nExiting...")
                break
            elif choice == 'new':
                print("\nStarting new game...")
                main()
                return
            
            # Capture current board
            print("\nCapturing current board...")
            current = capture_board(cam_idx, config)
            current_path = save_image(current, str(session_dir / f"{'white' if is_white_move else 'black'}_m{move_number}_current"))
            
            # Create diff image and analyze
            print("\nAnalyzing move...")
            diff = cv2.absdiff(initial, current)
            diff_path = save_image(diff, str(session_dir / f"{'white' if is_white_move else 'black'}_m{move_number}_diff"))
            
            # Analyze move from diff
            from_square, to_square = analyze_move_from_diff(diff, initial, current)
            print(f"\nDetected move: {from_square} -> {to_square}")
            
            # Update board state
            board_state = update_board_state(board_state, from_square.lower(), to_square.lower())
            print_board(board_state)
            
            # Update shadow board
            shadow_board = sync_shadow_board(board_state)
            fen_history.append(shadow_board.fen())
            
            # Save move data
            move_data = {
                "move_number": move_number,
                "player": "white" if is_white_move else "black",
                "timestamp": datetime.now().isoformat(),
                "from_square": from_square.lower(),
                "to_square": to_square.lower(),
                "current_image": str(Path(current_path).name),
                "fen": shadow_board.fen()
            }
            moves.append(move_data)
            
            # Save all data
            save_moves(session_dir, moves)
            save_board_states(session_dir, board_states)
            save_fen_history(session_dir, fen_history)
            
            print(shadow_board.fen())
            
            # Run Stockfish analysis every other move
            # if move_number % 2 == 0:
            print("\nRunning Stockfish analysis...")
            analysis = analyze_position_with_stockfish(shadow_board.fen())
            if analysis:
                print_stockfish_analysis(analysis, shadow_board)
            
            # Update for next move
            if not is_white_move:
                move_number += 1
            is_white_move = not is_white_move
            initial = current  # Use current image as next initial state
            
            print("\nImages saved successfully:")
            print(f"- Current board: {current_path}")
            print(f"- Diff image: {diff_path}")
            print(f"- Move detection visualization: {str(CAPTURES_DIR / 'move_detection.jpg')}")
            
            print("\nOptions:")
            print("1. Press Enter to continue to next move")
            print("2. Type 'new' to start a new game")
            print("3. Type 'exit' to quit")
            print("4. Type 'export' to export game positions")
            
            choice = input("\nYour choice: ").lower().strip()
            if choice == 'exit':
                print("\nExiting...")
                break
            elif choice == 'new':
                print("\nStarting new game...")
                main()
                return
            elif choice == 'export':
                export_file = export_game_positions(session_dir, fen_history)
                print(f"Game exported to {export_file}")
        
        print("\nProcess completed. Exiting...")
        sys.exit(0)
    
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 