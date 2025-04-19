"""
chess_vision_squares.py
---------------------
Capture and analyze chess moves using computer vision and square interpolation.
Uses calibrated board images divided into 64 squares for move detection.
"""

import cv2
import numpy as np
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from calibrate_board import load_config, apply_transform, calibrate_from_camera

# Setup
load_dotenv()
CAPTURES_DIR = Path("captures")
CAPTURES_DIR.mkdir(exist_ok=True)

# Board dimensions
BOARD_SIZE = (8, 8)  # 8x8 chessboard
SQUARE_SIZE = 100    # Size of each square in pixels

# Piece symbols for visualization
PIECE_SYMBOLS = {
    "white": {
        "pawn": "♙", "knight": "♘", "bishop": "♗",
        "rook": "♖", "queen": "♕", "king": "♔"
    },
    "black": {
        "pawn": "♟", "knight": "♞", "bishop": "♝",
        "rook": "♜", "queen": "♛", "king": "♚"
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


class ChessMove(BaseModel):
    """Represents a chess move."""
    move_notation: str
    from_square: str
    to_square: str
    piece_type: str
    piece_color: str
    captured_piece: Optional[str] = None

class SquareAnalyzer:
    """Analyzes chess moves by comparing board squares."""
    
    def __init__(self, board_size, square_size):
        """Initialize the analyzer with board size and square size."""
        self.board_size = board_size
        self.square_size = square_size
        self.square_cache = {}  # Cache for square templates
    
    def get_square_coordinates(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Get coordinates for all 64 squares."""
        height, width = img.shape[:2]
        square_w = width // self.board_size[1]
        square_h = height // self.board_size[0]
        
        squares = []
        for row in range(self.board_size[0]):
            for col in range(self.board_size[1]):
                x = col * square_w
                y = row * square_h
                squares.append((x, y, square_w, square_h))
        return squares
    
    def get_square_name(self, row: int, col: int) -> str:
        """Convert row/col to algebraic notation (e.g., 'e4')."""
        file = chr(ord('a') + col)
        rank = str(8 - row)
        return f"{file}{rank}"
    
    def get_square_content(self, img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Extract and preprocess a single square."""
        square = img[y:y+h, x:x+w]
        # Convert to grayscale
        gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
        # Apply threshold to reduce noise
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return thresh
    
    def compare_squares(self, sq1: np.ndarray, sq2: np.ndarray) -> float:
        """Compare two squares and return similarity score."""
        # Use structural similarity index
        return cv2.matchTemplate(sq1, sq2, cv2.TM_CCOEFF_NORMED)[0][0]
    
    def detect_changes(self, initial: np.ndarray, current: np.ndarray) -> Tuple[List[str], List[float]]:
        """Detect which squares changed between two board states."""
        changes = []
        scores = []
        squares = self.get_square_coordinates(initial)
        
        for idx, (x, y, w, h) in enumerate(squares):
            row, col = idx // self.board_size[1], idx % self.board_size[1]
            square_name = self.get_square_name(row, col)
            
            sq1 = self.get_square_content(initial, x, y, w, h)
            sq2 = self.get_square_content(current, x, y, w, h)
            
            similarity = self.compare_squares(sq1, sq2)
            if similarity < 0.8:  # Threshold for change detection
                changes.append(square_name)
                scores.append(similarity)
        
        return changes, scores
    
    def analyze_move(self, initial_path: str, current_path: str, board_state: Dict) -> ChessMove:
        """Analyze chess move by comparing squares."""
        # Read images (use processed/calibrated images if available)
        config = load_config()
        is_calibrated = config["camera"]["is_calibrated"] and config["calibration"]["matrix"] is not None
        
        if is_calibrated:
            initial_processed = str(Path(initial_path).parent / f"{Path(initial_path).stem}_processed.jpg")
            current_processed = str(Path(current_path).parent / f"{Path(current_path).stem}_processed.jpg")
            if Path(initial_processed).exists() and Path(current_processed).exists():
                initial_path = initial_processed
                current_path = current_processed
        
        initial = cv2.imread(initial_path)
        current = cv2.imread(current_path)
        
        # Detect changed squares
        changed_squares, scores = self.detect_changes(initial, current)
        
        if len(changed_squares) != 2:
            raise ValueError(f"Expected 2 changed squares, found {len(changed_squares)}")
        
        # Determine source and destination squares
        from_square = changed_squares[0]
        to_square = changed_squares[1]
        
        # Get piece information from board state
        piece = board_state.get(from_square)
        if not piece:
            raise ValueError(f"No piece found at square {from_square}")
        
        # Check for capture
        captured_piece = None
        if to_square in board_state:
            captured_piece = board_state[to_square]["type"]
        
        # Create move notation
        if piece["type"] == "pawn":
            move_notation = to_square
            if captured_piece:
                file_from = from_square[0]
                move_notation = f"{file_from}x{to_square}"
        else:
            piece_letter = piece["type"][0].upper()
            if piece_letter == "K":  # Knight uses 'N'
                piece_letter = "N"
            move_notation = f"{piece_letter}{to_square}"
            if captured_piece:
                move_notation = f"{piece_letter}x{to_square}"
        
        # Create visualization of the changes
        self.create_change_visualization(initial, current, from_square, to_square,
                                      str(Path(current_path).parent / f"{Path(current_path).stem}_changes.jpg"))
        
        return ChessMove(
            move_notation=move_notation,
            from_square=from_square,
            to_square=to_square,
            piece_type=piece["type"],
            piece_color=piece["color"],
            captured_piece=captured_piece
        )
    
    def create_change_visualization(self, initial: np.ndarray, current: np.ndarray,
                                  from_square: str, to_square: str, output_path: str):
        """Create visualization of detected changes."""
        vis = current.copy()
        squares = self.get_square_coordinates(current)
        
        # Find coordinates for source and destination squares
        for idx, (x, y, w, h) in enumerate(squares):
            row, col = idx // self.board_size[1], idx % self.board_size[1]
            square_name = self.get_square_name(row, col)
            
            if square_name == from_square:
                # Draw red rectangle for source
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(vis, "Source", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            elif square_name == to_square:
                # Draw green rectangle for destination
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(vis, "Destination", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save the visualization
        cv2.imwrite(output_path, vis)
        
        # Create and save diff image
        diff = cv2.absdiff(initial, current)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        
        # Add colored rectangles to diff image
        for idx, (x, y, w, h) in enumerate(squares):
            row, col = idx // self.board_size[1], idx % self.board_size[1]
            square_name = self.get_square_name(row, col)
            
            if square_name == from_square:
                cv2.rectangle(diff, (x, y), (x+w, y+h), (0, 0, 255), 2)
            elif square_name == to_square:
                cv2.rectangle(diff, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Save diff image with consistent naming
        diff_path = str(Path(output_path).parent / f"{Path(output_path).stem}_diff.jpg")
        cv2.imwrite(diff_path, diff)

def get_session_dir():
    """Create a new session directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = CAPTURES_DIR / f"game_{timestamp}"
    session_dir.mkdir(exist_ok=True)
    return session_dir

def save_board_states(session_dir, board_states):
    """Save board states history."""
    states_file = session_dir / "board.json"
    with open(states_file, 'w') as f:
        json.dump(board_states, f, indent=2)

def load_board_states(session_dir):
    """Load board states history."""
    states_file = session_dir / "board.json"
    if states_file.exists():
        with open(states_file, 'r') as f:
            return json.load(f)
    return [{
        "timestamp": datetime.now().isoformat(),
        "move_number": 0,
        "player": "initial",
        "board": INITIAL_BOARD
    }]

def save_moves(session_dir, moves):
    """Save move history."""
    moves_file = session_dir / "moves.json"
    with open(moves_file, 'w') as f:
        json.dump(moves, f, indent=2)

def update_board_state(board_state: Dict, move: Dict) -> Dict:
    """Update the board state based on a move."""
    new_state = board_state.copy()
    # Remove piece from old square
    if move["from_square"] in new_state:
        del new_state[move["from_square"]]
    # Add piece to new square
    new_state[move["to_square"]] = {
        "type": move["piece_type"],
        "color": move["piece_color"]
    }
    return new_state

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

def list_cameras(max_idx=10):
    """List available cameras."""
    cams = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok: cams.append(i)
            cap.release()
    return cams

def capture_board(camera_index, filename):
    """Capture and save chess board image."""
    # Load config
    config = load_config()
    
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {camera_index} not accessible.")
    
    # Set camera properties from config
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
    
    # Save original image
    cv2.imwrite(filename, frame)
    
    # If calibrated, apply transform
    if config["camera"]["is_calibrated"] and config["calibration"]["matrix"] is not None:
        matrix = np.array(config["calibration"]["matrix"])
        square_size = config["calibration"]["square_size"]
        processed_frame = apply_transform(frame, matrix, square_size)
        
        # Save processed image
        processed_path = str(Path(filename).parent / f"{Path(filename).stem}_processed.jpg")
        cv2.imwrite(processed_path, processed_frame)
        return processed_frame
    
    return frame

def select_chessboard(camera_index, config):
    """Select the chessboard area and create a uniform 8x8 grid."""
    print("\n=== Chessboard Selection ===")
    print("1. Click and drag to select the entire chessboard")
    print("2. Press 's' to save the selection")
    print("3. Press 'r' to reset")
    print("4. Press 'q' to quit")
    
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {camera_index} not accessible.")
    
    # Set camera properties from config
    settings = config["camera"]["settings"]
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, settings["exposure"])
    cap.set(cv2.CAP_PROP_BRIGHTNESS, settings["brightness"])
    cap.set(cv2.CAP_PROP_CONTRAST, settings["contrast"])
    cap.set(cv2.CAP_PROP_SATURATION, settings["saturation"])
    cap.set(cv2.CAP_PROP_GAIN, settings["gain"])
    
    # Variables for selection
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
    
    # Create window with WINDOW_NORMAL flag to allow resizing
    cv2.namedWindow('Chessboard Selection', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Chessboard Selection', draw_rectangle)
    
    def draw_chessboard_grid(frame, start_x, start_y, width, height):
        """Draw the chessboard grid with position labels."""
        # Calculate square size
        square_width = width // 8
        square_height = height // 8
        
        # Draw grid lines
        for i in range(9):
            # Draw horizontal lines
            cv2.line(frame, (start_x, start_y + i * square_height), 
                    (start_x + 8 * square_width, start_y + i * square_height), 
                    (0, 0, 255), 1)
            # Draw vertical lines
            cv2.line(frame, (start_x + i * square_width, start_y), 
                    (start_x + i * square_width, start_y + 8 * square_height), 
                    (0, 0, 255), 1)
        
        # Add position labels to each square
        for row in range(8):
            for col in range(8):
                # Calculate square position
                x = start_x + col * square_width
                y = start_y + row * square_height
                
                # Get square name (e.g., "a8", "h1")
                file = chr(ord('a') + col)
                rank = str(8 - row)
                square_name = f"{file}{rank}"
                
                # Draw square name in the center
                text_size = cv2.getTextSize(square_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = x + (square_width - text_size[0]) // 2
                text_y = y + (square_height + text_size[1]) // 2
                
                cv2.putText(frame, square_name, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add file labels (a-h)
        for i in range(8):
            file_label = chr(ord('a') + i)
            cv2.putText(frame, file_label, 
                      (start_x + i * square_width + square_width//2 - 5, 
                       start_y + 8 * square_height + 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add rank labels (8-1)
        for i in range(8):
            rank_label = str(8 - i)
            cv2.putText(frame, rank_label,
                      (start_x - 15, start_y + i * square_height + square_height//2 + 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Apply calibration transform if available
            if config["camera"]["is_calibrated"] and config["calibration"]["matrix"] is not None:
                matrix = np.array(config["calibration"]["matrix"])
                frame = apply_transform(frame, matrix, 100)  # Use default square size for transform
            
            # Draw the current selection
            if ix != -1 and iy != -1:
                if drawing:
                    # Show selection rectangle while dragging
                    cv2.rectangle(frame, (ix, iy), (fx, fy), (0, 255, 0), 2)
                elif fx != -1 and fy != -1 and board_width > 0 and board_height > 0:
                    # After dragging is complete, show the 8x8 grid
                    draw_chessboard_grid(frame, ix, iy, board_width, board_height)
            
            # Show instructions
            cv2.putText(frame, "Click and drag to select the chessboard", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 's' to save, 'r' to reset, 'q' to quit", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('Chessboard Selection', frame)
            
            # Wait for key press
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
                return None, None, None
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        if board_width > 0 and board_height > 0:
            # Calculate square size
            square_size = min(board_width, board_height) // 8
            return (ix, iy), (board_width, board_height), square_size
        else:
            return None, None, None
            
    except Exception as e:
        print(f"Error in chessboard selection: {e}")
        cap.release()
        cv2.destroyAllWindows()
        return None, None, None

def main():
    """Main execution loop."""
    try:
        # Create new session directory
        session_dir = get_session_dir()
        print(f"\nStarting new game session in {session_dir.name}")
        
        # Setup camera
        cams = list_cameras()
        if len(cams) < 2:
            sys.exit("Second camera not found. Please ensure at least 2 cameras are connected.")
        
        # Use the second camera (index 1)
        cam_idx = 1
        print(f"Using camera {cam_idx}")
        
        # Load config
        config = load_config()
        
        # Ask about calibration
        if not config["camera"]["is_calibrated"]:
            print("\nCamera is not calibrated.")
            calibrate = input("Would you like to calibrate now? (y/n): ").lower().strip() == 'y'
        else:
            calibrate = input("\nCamera is already calibrated. Recalibrate? (y/n): ").lower().strip() == 'y'
        
        if calibrate:
            print("\n=== Camera Calibration ===")
            print("Please ensure the chessboard is well-lit and clearly visible.")
            original_path, calibrated_path, _ = calibrate_from_camera(cam_idx, str(session_dir))
            
            if not calibrated_path:
                print("Calibration was cancelled or failed. Using default camera settings.")
            else:
                print("Camera successfully calibrated!")
                print(f"Calibration images saved to:\n- Original: {original_path}\n- Calibrated: {calibrated_path}")
        
        # Select chessboard and create grid
        print("\n=== Chessboard Selection ===")
        print("Please select the entire chessboard area.")
        board_pos, board_size, square_size = select_chessboard(cam_idx, config)
        
        if board_pos is None or board_size is None or square_size is None:
            print("Chessboard selection was cancelled. Using default values.")
            board_pos = (0, 0)
            board_size = (800, 800)
            square_size = 100
        
        print(f"\nSelected chessboard at position {board_pos} with size {board_size}")
        print(f"Square size: {square_size}px")
        
        # Initialize analyzer with selected board size
        analyzer = SquareAnalyzer((8, 8), square_size)
        print("Using square-based move detection")
        
        # Load existing moves and board states
        moves = []
        board_states = load_board_states(session_dir)
        board_state = board_states[-1]["board"].copy()  # Get latest state
        moves_file = session_dir / "moves.json"
        
        if moves_file.exists():
            with open(moves_file, 'r') as f:
                moves = json.load(f)
        
        move_number = len(moves) // 2 + 1  # Full moves (white + black)
        is_white_move = len(moves) % 2 == 0  # True if it's white's turn
        
        # Print initial board state
        print_board(board_state)
        
        # Capture initial board
        print("\nCapturing initial board...")
        initial_image = f"white_m{move_number}_initial.jpg" if is_white_move else f"black_m{move_number}_initial.jpg"
        capture_board(cam_idx, str(session_dir / initial_image))
        print("Initial board captured. Press Enter after opponent's move...")
        input()
        
        # Main game loop
        while True:
            # Capture current board
            print("\nCapturing current board...")
            current_image = f"white_m{move_number}_current.jpg" if is_white_move else f"black_m{move_number}_current.jpg"
            capture_board(cam_idx, str(session_dir / current_image))
            print("Current board captured.")
            
            try:
                # Analyze changes
                print("\nAnalyzing board state...")
                move = analyzer.analyze_move(
                    str(session_dir / initial_image),
                    str(session_dir / current_image),
                    board_state
                )
                
                print("\n── Analysis ──")
                print(f"Move Number: {move_number}")
                print(f"Player: {'White' if is_white_move else 'Black'}")
                print(f"Move: {move.move_notation}")
                print(f"From: {move.from_square}")
                print(f"To: {move.to_square}")
                if move.captured_piece:
                    print(f"Captured: {move.captured_piece} at {move.to_square}")
                
                # Update board state
                board_state = update_board_state(board_state, move.model_dump())
                
                # Save board state
                board_states.append({
                    "timestamp": datetime.now().isoformat(),
                    "move_number": move_number,
                    "player": "white" if is_white_move else "black",
                    "board": board_state
                })
                save_board_states(session_dir, board_states)
                
                # Print updated board state
                print_board(board_state)
                
                # Save move data
                move_data = {
                    "move_number": move_number,
                    "player": "white" if is_white_move else "black",
                    "timestamp": datetime.now().isoformat(),
                    "initial_image": initial_image,
                    "current_image": current_image,
                    "move_notation": move.move_notation,
                    "from_square": move.from_square,
                    "to_square": move.to_square,
                    "piece_type": move.piece_type,
                    "piece_color": move.piece_color,
                    "captured_piece": move.captured_piece
                }
                
                moves.append(move_data)
                save_moves(session_dir, moves)
                
            except ValueError as e:
                print(f"\nError analyzing move: {e}")
                print("Please try capturing the board again.")
                continue
            
            # Update for next move
            if not is_white_move:  # If black just moved, increment move number
                move_number += 1
            is_white_move = not is_white_move  # Switch turns
            initial_image = current_image
            
            # Prompt for next move or game end
            print("\nOptions:")
            print("1. Press Enter to continue to next move")
            print("2. Type 'cal' to recalibrate camera")
            print("3. Type 'new' to start a new game")
            print("4. Type 'exit' to quit")
            
            choice = input("\nYour choice: ").lower().strip()
            if choice == 'new':
                main()  # Start a new game
                return
            elif choice == 'exit':
                print("\nExiting...")
                sys.exit(0)
            elif choice == 'cal':
                print("\n=== Camera Recalibration ===")
                original_path, calibrated_path, _ = calibrate_from_camera(cam_idx, str(session_dir))
                if calibrated_path:
                    print("Camera successfully recalibrated!")
            # If Enter is pressed, continue with next move
    
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 