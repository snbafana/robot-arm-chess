"""
chess_vision.py
--------------
Capture and analyze chess moves using computer vision and Gemini AI.
"""

import cv2, base64, os, sys, json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv
from calibrate_board import calibrate_from_camera, load_config, apply_transform

# Setup
load_dotenv()
CAPTURES_DIR = Path("captures")
CAPTURES_DIR.mkdir(exist_ok=True)

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
    move_notation: str = Field(..., description="Standard algebraic notation for the move (e.g., e4, Nf6, Bxc4)")
    from_square: str = Field(..., description="Starting square (e.g., 'e2')")
    to_square: str = Field(..., description="Destination square (e.g., 'e4')")
    piece_type: str = Field(..., description="Type of piece that moved (pawn, knight, bishop, rook, queen, king)")
    piece_color: str = Field(..., description="Color of piece that moved (white or black)")
    captured_piece: Optional[str] = Field(None, description="Type of piece captured, if any")

class ChessAnalyzer:
    """Analyzes chess moves using Gemini AI."""
    
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    def analyze_move(self, first_path: str, next_path: str, board_state: Dict) -> ChessMove:
        """Analyze chess move using Gemini's vision model."""
        prompt = """Given two images showing a chess position before and after a move, give the move in the json format provided."""
        
        # Upload both images
        first_file = self.client.files.upload(file=first_path)
        next_file = self.client.files.upload(file=next_path)
        
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                prompt,
                first_file,
                next_file
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": ChessMove,
            }
        )
        
        return response.parsed

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
        
        # Initialize analyzer
        analyzer = ChessAnalyzer()
        print("Using Gemini for analysis")
        
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
        
        # Capture initial base position
        print("\nCapturing initial base position...")
        base_image = "base.jpg"
        capture_board(cam_idx, str(session_dir / base_image))
        print("Base position captured. Press Enter after opponent's move...")
        input()
        
        # Main game loop
        while True:
            # Capture opponent's move
            print("\nCapturing opponent's move...")
            move_image = f"white_m{move_number}.jpg" if is_white_move else f"black_m{move_number}.jpg"
            capture_board(cam_idx, str(session_dir / move_image))
            print("Move captured. Analyzing...")
            
            try:
                # Analyze move
                print("\nAnalyzing move with Gemini...")
                move = analyzer.analyze_move(
                    str(session_dir / base_image),
                    str(session_dir / move_image),
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
                    "base_image": base_image,
                    "move_image": move_image,
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
            
            # Set current move as base for next move
            base_image = move_image
            
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