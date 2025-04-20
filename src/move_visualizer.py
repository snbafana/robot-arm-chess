"""
move_visualizer.py
-----------------
Visualize chess moves on the processed board image.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, Optional

def load_config() -> Dict:
    """Load configuration from config file."""
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

def chess_to_pixel(square: str, board_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    Convert chess notation (e.g., 'a1') to pixel coordinates on the processed board image.
    
    Args:
        square: Chess notation (e.g., 'a1')
        board_size: Tuple of (width, height) of the processed board image
    
    Returns:
        Tuple of (x, y) pixel coordinates
    """
    width, height = board_size
    square_width = width // 8
    square_height = height // 8
    
    # Convert chess notation to grid coordinates
    file = ord(square[0].lower()) - ord('a')  # a=0, b=1, ..., h=7
    rank = int(square[1]) - 1  # 1=0, 2=1, ..., 8=7
    
    # Calculate center of the square
    x = file * square_width + square_width // 2
    y = (7 - rank) * square_height + square_height // 2  # Flip y-axis for chess notation
    
    return (x, y)

def visualize_move(from_square: str, to_square: str, board_image: np.ndarray) -> np.ndarray:
    """
    Visualize a chess move on the processed board image.
    
    Args:
        from_square: Starting square in chess notation (e.g., 'a1')
        to_square: Destination square in chess notation (e.g., 'a3')
        board_image: The processed board image from chess_vision_squares.py
    
    Returns:
        Image with move visualization
    """
    # Create a copy of the image to draw on
    vis_image = board_image.copy()
    
    # Get board dimensions
    height, width = vis_image.shape[:2]
    
    # Get pixel coordinates
    start_point = chess_to_pixel(from_square, (width, height))
    end_point = chess_to_pixel(to_square, (width, height))
    
    # Draw points and line
    cv2.circle(vis_image, start_point, 10, (0, 0, 255), -1)  # Red dot for start
    cv2.circle(vis_image, end_point, 10, (255, 0, 0), -1)    # Blue dot for end
    cv2.line(vis_image, start_point, end_point, (0, 255, 0), 2)  # Green line
    
    # Add text labels
    cv2.putText(vis_image, from_square, (start_point[0] + 15, start_point[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(vis_image, to_square, (end_point[0] + 15, end_point[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return vis_image

def main():
    """Test the visualization with example moves."""
    # Load configuration
    config = load_config()
    if not config or not config["camera"]["is_calibrated"]:
        print("Error: Board not calibrated. Please run chess_vision_squares.py first.")
        return
    
    # Get example squares
    from_square = input("Enter starting square (e.g., a1): ").strip()
    to_square = input("Enter destination square (e.g., a3): ").strip()
    
    # Load the last captured board image
    captures_dir = Path("captures")
    if not captures_dir.exists():
        print("Error: No captures directory found. Please run chess_vision_squares.py first.")
        return
    
    # Find the most recent game directory
    game_dirs = sorted(captures_dir.glob("game_*"), reverse=True)
    if not game_dirs:
        print("Error: No game captures found. Please run chess_vision_squares.py first.")
        return
    
    # Find the most recent current board image
    current_images = list(game_dirs[0].glob("*_current.jpg"))
    if not current_images:
        print("Error: No current board image found. Please run chess_vision_squares.py first.")
        return
    
    # Load the board image
    board_image = cv2.imread(str(current_images[0]))
    if board_image is None:
        print("Error: Could not load board image.")
        return
    
    # Visualize the move
    vis_image = visualize_move(from_square, to_square, board_image)
    
    # Show the result
    cv2.imshow("Move Visualization", vis_image)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 