"""
calibrate_board.py
-----------------
Manual calibration tool for chessboard corner selection.
Allows user to click on the four corners of the chessboard to get proper perspective transform.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime

# Global variables for corner selection
corners = []
image_copy = None
window_name = "Manual Calibration"

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
            "square_size": 100,
            "matrix": None,
            "last_calibration": None
        }
    }

def save_config(config: Dict):
    """Save configuration to config file."""
    with open("config.json", 'w') as f:
        json.dump(config, f, indent=4)

def apply_transform(image: np.ndarray, matrix: np.ndarray, square_size: int) -> np.ndarray:
    """Apply perspective transform using saved matrix."""
    board_width = 8 * square_size
    board_height = 8 * square_size
    return cv2.warpPerspective(image, matrix, (board_width, board_height))

def click_event(event, x, y, flags, params):
    """Mouse click callback function."""
    global corners, image_copy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(corners) < 4:
            corners.append((x, y))
            # Draw point
            cv2.circle(image_copy, (x, y), 5, (0, 255, 0), -1)
            # Draw number
            cv2.putText(image_copy, str(len(corners)), (x+10, y+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(window_name, image_copy)

def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order corners in [top-left, top-right, bottom-right, bottom-left] order."""
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

def manual_calibrate(image: np.ndarray, square_size: int = 100) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Manually calibrate chessboard corners using mouse clicks.
    
    Args:
        image: Input image
        square_size: Size of each square in pixels for the output image
        
    Returns:
        Tuple of (success, warped_image, transform_matrix)
    """
    global corners, image_copy, window_name
    
    # Reset corners
    corners = []
    # Make a copy of the image
    image_copy = image.copy()
    
    # Create window and set mouse callback
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)
    
    # Instructions
    print("\nManual Calibration Instructions:")
    print("1. Click on the four corners of the chessboard in this order:")
    print("   - Top-left corner")
    print("   - Top-right corner")
    print("   - Bottom-right corner")
    print("   - Bottom-left corner")
    print("2. Press 'r' to reset if you make a mistake")
    print("3. Press 'c' to confirm when all corners are selected")
    print("4. Press 'q' to quit\n")
    
    while True:
        cv2.imshow(window_name, image_copy)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):  # Reset
            corners = []
            image_copy = image.copy()
            cv2.imshow(window_name, image_copy)
        
        elif key == ord('c') and len(corners) == 4:  # Confirm
            break
        
        elif key == ord('q'):  # Quit
            cv2.destroyAllWindows()
            return False, None, None
    
    cv2.destroyAllWindows()
    
    # Convert corners to numpy array and order them
    corners_array = np.array(corners, dtype="float32")
    ordered_corners = order_corners(corners_array)
    
    # Calculate target dimensions
    board_width = 8 * square_size   # 8 squares wide
    board_height = 8 * square_size  # 8 squares tall
    
    # Define target points
    dst_points = np.array([
        [0, 0],
        [board_width - 1, 0],
        [board_width - 1, board_height - 1],
        [0, board_height - 1]
    ], dtype="float32")
    
    # Calculate perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, dst_points)
    
    # Apply perspective transform
    warped = cv2.warpPerspective(image, matrix, (board_width, board_height))
    
    return True, warped, matrix

def save_calibration(image: np.ndarray, output_dir: Path, prefix: str = "calibrated") -> Tuple[str, str, Optional[np.ndarray]]:
    """
    Save both original and calibrated images.
    
    Args:
        image: Input image
        output_dir: Directory to save images
        prefix: Prefix for saved files
        
    Returns:
        Tuple of (original_path, calibrated_path, transform_matrix)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save original image
    original_path = str(output_dir / f"{prefix}_original.jpg")
    cv2.imwrite(original_path, image)
    
    # Load config
    config = load_config()
    square_size = config["calibration"]["square_size"]
    
    # Perform manual calibration
    success, calibrated, matrix = manual_calibrate(image, square_size)
    
    if success:
        # Save calibrated image
        calibrated_path = str(output_dir / f"{prefix}_calibrated.jpg")
        cv2.imwrite(calibrated_path, calibrated)
        
        # Update config with new calibration
        config["camera"]["is_calibrated"] = True
        config["calibration"]["matrix"] = matrix.tolist()
        config["calibration"]["last_calibration"] = datetime.now().isoformat()
        save_config(config)
        
        print(f"\nSaved calibrated image to: {calibrated_path}")
        return original_path, calibrated_path, matrix
    else:
        print("\nCalibration cancelled")
        return original_path, "", None

def calibrate_from_camera(camera_index: int = 0, output_dir: str = "calibration") -> Tuple[str, str, Optional[np.ndarray]]:
    """
    Capture image from camera and perform manual calibration.
    
    Args:
        camera_index: Index of the camera to use
        output_dir: Directory to save images
        
    Returns:
        Tuple of (original_path, calibrated_path, transform_matrix)
    """
    # Load config
    config = load_config()
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {camera_index} not accessible")
    
    # Set camera properties from config
    settings = config["camera"]["settings"]
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, settings["exposure"])
    cap.set(cv2.CAP_PROP_BRIGHTNESS, settings["brightness"])
    cap.set(cv2.CAP_PROP_CONTRAST, settings["contrast"])
    cap.set(cv2.CAP_PROP_SATURATION, settings["saturation"])
    cap.set(cv2.CAP_PROP_GAIN, settings["gain"])
    
    print("\nPress SPACE to capture image or 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.imshow("Camera Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space key
            cv2.destroyWindow("Camera Feed")
            cap.release()
            return save_calibration(frame, output_dir)
        
        elif key == ord('q'):  # Quit
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return "", "", None

def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manual chessboard calibration tool")
    parser.add_argument("--camera", type=int, default=1,
                       help="Camera index to use (default: 1)")
    parser.add_argument("--image", type=str,
                       help="Path to image file (optional, uses camera if not provided)")
    parser.add_argument("--output", type=str, default="calibration",
                       help="Output directory (default: calibration)")
    
    args = parser.parse_args()
    
    if args.image:
        # Calibrate from image file
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not read image file: {args.image}")
            return
        save_calibration(image, args.output)
    else:
        # Calibrate from camera
        calibrate_from_camera(args.camera, args.output)

if __name__ == "__main__":
    main() 