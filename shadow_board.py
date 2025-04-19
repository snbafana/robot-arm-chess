"""
shadow_board.py
--------------
Chess board state management and python-chess integration.
Handles board state tracking, FEN conversion, and move validation.
"""
import chess
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def sync_shadow_board(board_state: Dict) -> chess.Board:
    """Create a chess.Board object from the board_state dictionary."""
    # Create an empty board
    shadow_board = chess.Board.empty()
    
    # Piece type mapping from string to chess library constants
    piece_type_map = {
        "pawn": chess.PAWN,
        "knight": chess.KNIGHT,
        "bishop": chess.BISHOP,
        "rook": chess.ROOK,
        "queen": chess.QUEEN,
        "king": chess.KING
    }
    
    # Add pieces from board_state
    for square_name, piece_info in board_state.items():
        # Convert square name to square index (e.g., "e4" -> chess.E4)
        square = chess.parse_square(square_name)
        
        # Get piece type and color
        piece_type = piece_type_map[piece_info["type"]]
        color = chess.WHITE if piece_info["color"] == "white" else chess.BLACK
        
        # Create piece and place it on the board
        piece = chess.Piece(piece_type, color)
        shadow_board.set_piece_at(square, piece)
    
    # Set turn, castling rights, and en passant based on game state
    # Default to white's turn if not specified elsewhere
    shadow_board.turn = chess.WHITE
    
    return shadow_board

def save_fen_history(session_dir, fen_list):
    """Save game history in FEN notation."""
    fen_file = session_dir / "fen_history.json"
    with open(fen_file, 'w') as f:
        json.dump(fen_list, f, indent=2)

def load_fen_history(session_dir):
    """Load game history in FEN notation."""
    fen_file = session_dir / "fen_history.json"
    if fen_file.exists():
        with open(fen_file, 'r') as f:
            return json.load(f)
    return [chess.Board().fen()]  # Start with the initial position

def export_game_positions(session_dir, fen_list):
    """Export all game positions in FEN for later analysis."""
    export_file = session_dir / "game_positions.txt"
    
    with open(export_file, "w") as f:
        for i, fen in enumerate(fen_list):
            # Write move number and FEN
            move_num = i // 2 + 1 if i > 0 else 0
            player = "Initial" if i == 0 else ("White" if i % 2 == 1 else "Black")
            move_text = f"Initial position" if i == 0 else f"Move {move_num}: {player}"
            
            f.write(f"{move_text}\n")
            f.write(f"FEN: {fen}\n")
            
            # Create a board from FEN to show visual representation
            board = chess.Board(fen)
            f.write(f"{board}\n\n")
    
    print(f"\nGame positions exported to: {export_file}")
    return export_file 