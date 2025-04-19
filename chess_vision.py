"""
chess_vision.py
--------------
Capture and analyze chess moves using computer vision and AI with structured outputs.
Supports both OpenAI and Gemini models.
"""
import cv2, base64, os, sys, json, chess
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal, Dict, List, Tuple, Union
from pydantic import BaseModel, Field
from openai import OpenAI
from google import genai
import instructor
from dotenv import load_dotenv
from calibrate_board import calibrate_from_camera, load_config, apply_transform
import websocket

# Import shadow board and stockfish modules
from shadow_board import sync_shadow_board, save_fen_history, load_fen_history, export_game_positions
from stockfish_api import analyze_position_with_stockfish, print_stockfish_analysis

# Setup
load_dotenv()
CAPTURES_DIR = Path("captures")
CAPTURES_DIR.mkdir(exist_ok=True)

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

# Board dimensions
BOARD_SIZE = (8, 8)  # 8x8 chessboard
SQUARE_SIZE = 100    # Target size for each square in pixels

def find_chessboard_corners(image: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
    """Find chessboard corners in the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)
    
    if ret:
        # Refine corner detection
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Get the four corner points of the board
        board_corners = np.float32([
            corners[0][0],                    # Top-left
            corners[BOARD_SIZE[0]-1][0],      # Top-right
            corners[-1][0],                   # Bottom-right
            corners[-BOARD_SIZE[0]][0]        # Bottom-left
        ])
        
        return True, corners, board_corners
    
    return False, None, None

def order_points(pts: np.ndarray) -> np.ndarray:
    """Order points in top-left, top-right, bottom-right, bottom-left order."""
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left will have smallest sum
    # Bottom-right will have largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right will have smallest difference
    # Bottom-left will have largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def perspective_transform(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Apply perspective transform to get a top-down view of the board."""
    # Order the corners
    corners = order_points(corners)
    
    # Compute target dimensions
    board_width = BOARD_SIZE[0] * SQUARE_SIZE
    board_height = BOARD_SIZE[1] * SQUARE_SIZE
    
    # Define target points
    dst_points = np.array([
        [0, 0],
        [board_width - 1, 0],
        [board_width - 1, board_height - 1],
        [0, board_height - 1]
    ], dtype="float32")
    
    # Calculate perspective transform matrix
    matrix = cv2.getPerspectiveTransform(corners, dst_points)
    
    # Apply perspective transform
    warped = cv2.warpPerspective(image, matrix, (board_width, board_height))
    
    return warped

def preprocess_board_image(image_input: Union[str, np.ndarray], output_path: str = None) -> Tuple[bool, np.ndarray]:
    """Preprocess chess board image to get aligned top-down view.
    
    Args:
        image_input: Either a path to an image file (str) or a numpy array containing the image
        output_path: Optional path to save the processed image
    """
    # Handle input
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            return False, None
    else:
        image = image_input
    
    # Find chessboard corners
    ret, corners, board_corners = find_chessboard_corners(image)
    if not ret:
        return False, None
    
    # Draw detected corners for visualization
    debug_image = image.copy()
    cv2.drawChessboardCorners(debug_image, BOARD_SIZE, corners, ret)
    
    # Save debug image showing corner detection
    if output_path:
        debug_path = str(Path(output_path).parent / f"{Path(output_path).stem}_corners.jpg")
        cv2.imwrite(debug_path, debug_image)
    
    # Apply perspective transform
    warped = perspective_transform(image, board_corners)
    
    # Save processed image if output path provided
    if output_path:
        cv2.imwrite(output_path, warped)
    
    return True, warped

def get_session_dir():
    """Create a new session directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = CAPTURES_DIR / f"game_{timestamp}"
    session_dir.mkdir(exist_ok=True)
    return session_dir

def load_moves(session_dir):
    """Load move history and board state."""
    moves = []
    board_state = INITIAL_BOARD.copy()
    moves_file = session_dir / "moves.json"
    
    if moves_file.exists():
        with open(moves_file, 'r') as f:
            moves = json.load(f)
            # Reconstruct board state from moves
            for move in moves:
                # Remove piece from old square
                if move["from_square"] in board_state:
                    del board_state[move["from_square"]]
                # Add piece to new square
                board_state[move["to_square"]] = {
                    "type": move["piece_type"],
                    "color": move["piece_color"]
                }
                # Handle captures
                if move["captured_piece"]:
                    # The captured piece is already removed from the board state
                    # as it was overwritten by the moving piece
                    pass
    
    # Create shadow_board from board_state
    shadow_board = sync_shadow_board(board_state)
    
    # If moves exist, try to determine whose turn it is
    if moves and len(moves) > 0:
        last_move = moves[-1]
        # Set turn to opposite of last move's color
        shadow_board.turn = chess.BLACK if last_move["piece_color"] == "white" else chess.WHITE
    
    return moves, board_state, shadow_board

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

def create_diff_image(initial_path: str, current_path: str, output_path: str) -> str:
    """Create a difference image highlighting changes between two board states."""
    # Load config to check calibration status
    config = load_config()
    is_calibrated = config["camera"]["is_calibrated"] and config["calibration"]["matrix"] is not None
    
    # Read images
    initial = cv2.imread(initial_path)
    current = cv2.imread(current_path)
    
    # If calibrated, use the processed images instead
    if is_calibrated:
        initial_processed = str(Path(initial_path).parent / f"{Path(initial_path).stem}_processed.jpg")
        current_processed = str(Path(current_path).parent / f"{Path(current_path).stem}_processed.jpg")
        if Path(initial_processed).exists() and Path(current_processed).exists():
            initial = cv2.imread(initial_processed)
            current = cv2.imread(current_processed)
    
    # Ensure images are the same size
    if initial.shape != current.shape:
        current = cv2.resize(current, (initial.shape[1], initial.shape[0]))
    
    # Convert to grayscale
    initial_gray = cv2.cvtColor(initial, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    diff = cv2.absdiff(initial_gray, current_gray)
    
    # Apply threshold to highlight significant changes
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Find contours of changes
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Create visualization
    diff_vis = current.copy()
    
    if len(contours) >= 2:
        # Get the two largest changes (source and destination squares)
        source_contour = contours[0]
        dest_contour = contours[1]
        
        # Draw rectangles: red for source, green for destination
        x1, y1, w1, h1 = cv2.boundingRect(source_contour)
        x2, y2, w2, h2 = cv2.boundingRect(dest_contour)
        
        # Draw source square in red
        cv2.rectangle(diff_vis, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)  # Red
        cv2.putText(diff_vis, "Source", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw destination square in green
        cv2.rectangle(diff_vis, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)  # Green
        cv2.putText(diff_vis, "Destination", (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        # If we can't find two distinct changes, draw all changes in yellow
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small changes
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(diff_vis, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow
    
    # Add legend
    cv2.putText(diff_vis, "Red: Source Square", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(diff_vis, "Green: Destination Square", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Save the visualization
    cv2.imwrite(output_path, diff_vis)
    return output_path

class ChessMove(BaseModel):
    """Represents a chess move."""
    move_notation: str = Field(..., description="Standard algebraic notation for the move (e.g., e4, Nf6, Bxc4)")
    from_square: str = Field(..., description="Starting square (e.g., 'e2')")
    to_square: str = Field(..., description="Destination square (e.g., 'e4')")
    piece_type: str = Field(..., description="Type of piece that moved (pawn, knight, bishop, rook, queen, king)")
    piece_color: str = Field(..., description="Color of piece that moved (white or black)")
    captured_piece: Optional[str] = Field(None, description="Type of piece captured, if any")

class ChessAnalyzer:
    """Unified interface for chess analysis using different AI providers."""
    
    def __init__(self, provider: Literal["openai", "gemini"] = "openai"):
        self.provider = provider
        if provider == "openai":
            self.client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        else:
            self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    def analyze_move(self, initial_path: str, current_path: str, board_state: Dict) -> ChessMove:
        """Analyze chess move using the selected provider."""
        # Create diff image with consistent naming
        session_dir = Path(initial_path).parent
        move_prefix = Path(current_path).stem.split('_')[0:2]  # Get 'white_m1' or 'black_m1' part
        diff_path = str(session_dir / f"{'_'.join(move_prefix)}_diff.jpg")
        
        # Check if we should use processed images for analysis
        config = load_config()
        is_calibrated = config["camera"]["is_calibrated"] and config["calibration"]["matrix"] is not None
        
        if is_calibrated:
            initial_processed = str(Path(initial_path).parent / f"{Path(initial_path).stem}_processed.jpg")
            current_processed = str(Path(current_path).parent / f"{Path(current_path).stem}_processed.jpg")
            if Path(initial_processed).exists() and Path(current_processed).exists():
                initial_path = initial_processed
                current_path = current_processed
        
        create_diff_image(initial_path, current_path, diff_path)
        
        if self.provider == "openai":
            return self._analyze_openai(initial_path, current_path, diff_path, board_state)
        else:
            return self._analyze_gemini(initial_path, current_path, diff_path, board_state)
    
    def _analyze_openai(self, initial_path: str, current_path: str, diff_path: str, board_state: Dict) -> ChessMove:
        """Analyze using OpenAI's vision model to detect the move made."""
        base64_initial = self._encode_image(initial_path)
        base64_current = self._encode_image(current_path)
        base64_diff = self._encode_image(diff_path)
        
        prompt = """Given the current board state and three images showing a chess position before and after a move,
        determine the chess move that was made. The first image shows the position before the move,
        the second shows the position after the move, and the third shows a visualization of the changes.
        
        In the difference visualization:
        - RED rectangle marks the SOURCE square (where the piece moved from)
        - GREEN rectangle marks the DESTINATION square (where the piece moved to)
        
        Current board state:
        {board_state}
        
        Return ONLY the move in standard algebraic notation and its details.
        Focus on:
        1. Which piece moved (type and color) - look at the RED (source) square in the initial position
        2. From which square to which square - use the RED (source) and GREEN (destination) markers
        3. If any piece was captured - check if there was a piece on the GREEN (destination) square
        4. The move in standard algebraic notation (e.g., e4, Nf6, Bxc4)""".format(
            board_state=json.dumps(board_state, indent=2)
        )
        
        response = self.client.chat.completions.create(
            model="gpt-4.1",
            response_model=ChessMove,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_initial}"}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_current}"}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_diff}"}
                        }
                    ]
                }
            ]
        )
        return response
    
    def _analyze_gemini(self, initial_path: str, current_path: str, diff_path: str, board_state: Dict) -> ChessMove:
        """Analyze using Gemini's vision model to detect the move made."""
        prompt = """Given the current board state and three images showing a chess position before and after a move,
        determine the chess move that was made. The first image shows the position before the move,
        the second shows the position after the move, and the third shows a visualization of the changes.
        
        In the difference visualization:
        - RED rectangle marks the SOURCE square (where the piece moved from)
        - GREEN rectangle marks the DESTINATION square (where the piece moved to)
        
        RELY HEAVILY ON THE RED AND GREEN INDICATORS TO DETERMINE THE EXACT SQUARES INVOLVED.
        
        Current board state:
        {board_state}
        
        Return ONLY the move in standard algebraic notation and its details.
        Focus on:
        1. Which piece moved (type and color) - look at the RED (source) square in the initial position
        2. From which square to which square - use the RED (source) and GREEN (destination) markers
        3. If any piece was captured - check if there was a piece on the GREEN (destination) square
        4. The move in standard algebraic notation (e.g., e4, Nf6, Bxc4)""".format(
            board_state=json.dumps(board_state, indent=2)
        )
        
        # Load images
        with open(initial_path, "rb") as f:
            initial_image = f.read()
        with open(current_path, "rb") as f:
            current_image = f.read()
        with open(diff_path, "rb") as f:
            diff_image = f.read()
        
        response = self.client.models.generate_content(
            model="gemini-2.5-flash-preview-04-17",
            contents=[
                {"role": "user", "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": initial_image}},
                    {"inline_data": {"mime_type": "image/jpeg", "data": current_image}},
                    {"inline_data": {"mime_type": "image/jpeg", "data": diff_image}}
                ]}
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": ChessMove,
            }
        )
        
        return response.parsed
    
    def _encode_image(self, image_path: str) -> str:
        """Convert image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

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
            from calibrate_board import calibrate_from_camera
            original_path, calibrated_path, _ = calibrate_from_camera(cam_idx, str(session_dir))
            
            if not calibrated_path:
                print("Calibration was cancelled or failed. Using default camera settings.")
            else:
                print("Camera successfully calibrated!")
                print(f"Calibration images saved to:\n- Original: {original_path}\n- Calibrated: {calibrated_path}")
        
        # Choose AI provider
        provider = input("\nChoose AI provider (openai/gemini): ").lower()
        while provider not in ["openai", "gemini"]:
            provider = input("Please choose either 'openai' or 'gemini': ").lower()
        
        analyzer = ChessAnalyzer(provider=provider)
        print(f"Using {provider.upper()} for analysis")
        
        # Load existing moves and board states
        moves, board_state, shadow_board = load_moves(session_dir)
        board_states = load_board_states(session_dir)
        fen_history = load_fen_history(session_dir)
        
        if not moves:
            # If no moves were loaded, initialize FEN history with current position
            fen_history = [shadow_board.fen()]
        
        move_number = len(moves) // 2 + 1  # Full moves (white + black)
        is_white_move = len(moves) % 2 == 0  # True if it's white's turn
        
        # Print initial board state
        print_board(board_state)
        print("\nCurrent board position (FEN notation):")
        print(shadow_board.fen())
        print("\nCurrent board position (visual):")
        print(shadow_board)
        
        # Initial Stockfish analysis
        print("\nAnalyzing initial position with Stockfish...")
        stockfish_analysis = analyze_position_with_stockfish(shadow_board.fen())
        if stockfish_analysis:
            print_stockfish_analysis(stockfish_analysis, shadow_board)
        
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
            
            # Analyze changes
            print(f"\nAnalyzing board state with {provider.upper()}...")
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
            
            # Update shadow_board with the detected move
            try:
                # Try to convert the move to UCI format for the chess library
                from_square = chess.parse_square(move.from_square)
                to_square = chess.parse_square(move.to_square)
                
                # Check for special moves
                chess_move = None
                
                # Check for castling (king moves two squares)
                if move.piece_type == "king":
                    # Kingside castling
                    if move.from_square == "e1" and move.to_square == "g1":
                        chess_move = chess.Move.from_uci("e1g1")  # White kingside
                    elif move.from_square == "e8" and move.to_square == "g8":
                        chess_move = chess.Move.from_uci("e8g8")  # Black kingside
                    # Queenside castling
                    elif move.from_square == "e1" and move.to_square == "c1":
                        chess_move = chess.Move.from_uci("e1c1")  # White queenside
                    elif move.from_square == "e8" and move.to_square == "c8":
                        chess_move = chess.Move.from_uci("e8c8")  # Black queenside
                
                # Check for pawn promotion
                if move.piece_type in ["queen", "rook", "bishop", "knight"] and move.move_notation and "=" in move.move_notation:
                    # Get promotion piece from notation
                    promotion_piece = move.move_notation.split("=")[1][0].lower()
                    promotion_map = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}
                    if promotion_piece in promotion_map:
                        chess_move = chess.Move(from_square, to_square, promotion=promotion_map[promotion_piece])
                
                # Default move if not a special move
                if chess_move is None:
                    chess_move = chess.Move(from_square, to_square)
                
                # Try the parsed move first
                if chess_move in shadow_board.legal_moves:
                    shadow_board.push(chess_move)
                    print("\nMove applied to shadow_board successfully!")
                else:
                    # If not successful, try to parse the SAN notation if available
                    try:
                        if move.move_notation:
                            san_move = shadow_board.parse_san(move.move_notation)
                            shadow_board.push(san_move)
                            print("\nMove applied from SAN notation successfully!")
                        else:
                            raise ValueError("No SAN notation available")
                    except ValueError:
                        # As a last resort, try to find a legal move between the squares
                        legal_moves = list(shadow_board.legal_moves)
                        matching_moves = [m for m in legal_moves if 
                                        m.from_square == from_square and 
                                        m.to_square == to_square]
                        
                        if matching_moves:
                            shadow_board.push(matching_moves[0])
                            print("\nMove applied from matching squares!")
                        else:
                            # If all else fails, resync the board
                            print("\nWarning: Move not legal in shadow_board, resyncing...")
                            shadow_board = sync_shadow_board(board_state)
                            # Set the correct turn
                            shadow_board.turn = chess.BLACK if is_white_move else chess.WHITE
                
                print("\nCurrent board position (FEN notation):")
                print(shadow_board.fen())
                
                print("\nCurrent board position (visual):")
                print(shadow_board)
                
                # Add FEN to history
                fen_history.append(shadow_board.fen())
                save_fen_history(session_dir, fen_history)
                
                # Check for game-ending conditions
                if shadow_board.is_checkmate():
                    winner = "Black" if shadow_board.turn == chess.WHITE else "White"
                    print(f"\n🏆 CHECKMATE! {winner} wins!")
                elif shadow_board.is_stalemate():
                    print("\n🤝 STALEMATE! Game is drawn.")
                elif shadow_board.is_insufficient_material():
                    print("\n🤝 DRAW DUE TO INSUFFICIENT MATERIAL!")
                elif shadow_board.can_claim_fifty_moves():
                    print("\n🤝 FIFTY-MOVE RULE can be claimed.")
                elif shadow_board.can_claim_threefold_repetition():
                    print("\n🤝 THREEFOLD REPETITION can be claimed.")
                
                if shadow_board.is_check():
                    print("\n⚠️ CHECK!")
                
                # Get Stockfish analysis for the current position
                print("\nAnalyzing position with Stockfish...")
                stockfish_analysis = analyze_position_with_stockfish(shadow_board.fen())
                if stockfish_analysis:
                    print_stockfish_analysis(stockfish_analysis, shadow_board)
                
            except Exception as e:
                print(f"\nError updating shadow_board: {e}")
                # Resync as a fallback
                shadow_board = sync_shadow_board(board_state)
            
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
                "provider": provider,
                "move_notation": move.move_notation,
                "from_square": move.from_square,
                "to_square": move.to_square,
                "piece_type": move.piece_type,
                "piece_color": move.piece_color,
                "captured_piece": move.captured_piece
            }
            
            moves.append(move_data)
            save_moves(session_dir, moves)
            
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
            print("5. Type 'export' to export the game positions in FEN notation")
            
            choice = input("\nYour choice: ").lower().strip()
            if choice == 'new':
                main()  # Start a new game
            elif choice == 'exit':
                print("\nExiting...")
                sys.exit(0)
            elif choice == 'cal':
                print("\n=== Camera Recalibration ===")
                original_path, calibrated_path, _ = calibrate_from_camera(cam_idx, str(session_dir))
                if calibrated_path:
                    print("Camera successfully recalibrated!")
            elif choice == 'export':
                export_file = export_game_positions(session_dir, fen_history)
                print(f"Game exported to {export_file}")
            # If Enter is pressed, continue with next move
        
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)

if __name__ == "__main__":
    main() 