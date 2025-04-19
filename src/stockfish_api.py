"""
stockfish_api.py
---------------
Stockfish chess engine API integration.
Provides functions for analyzing chess positions using Stockfish via WebSocket.
"""
import websocket
import json
import threading
import uuid
import time
import chess
from typing import Dict, Optional

def analyze_position_with_stockfish(fen: str, depth: int = 12) -> Optional[Dict]:
    """
    Analyze a chess position using the Stockfish API.
    Simple function version that doesn't maintain a persistent connection.
    
    Args:
        fen: FEN notation of the position
        depth: Analysis depth (1-18)
        
    Returns:
        Dictionary with analysis results or None if failed
    """
    try:
        # Create WebSocket connection
        ws = websocket.create_connection('wss://chess-api.com/v1')
        
        # Prepare and send the request
        request = {
            'fen': fen,
            'depth': min(18, depth)
        }
        ws.send(json.dumps(request))
        
        # Wait for response(s)
        best_move = None
        wait_time = 0
        max_wait = 10  # Maximum wait time in seconds
        
        while wait_time < max_wait:
            response = ws.recv()
            data = json.loads(response)
            
            # If this is a "bestmove" response, use it
            if data.get('type') == 'bestmove':
                best_move = data
                break
            
            # Otherwise keep the latest response
            best_move = data
            
            # If we've reached the desired depth, break
            if data.get('depth') >= depth:
                break
                
            time.sleep(0.5)
            wait_time += 0.5
        
        # Close connection
        ws.close()
        
        return best_move
    
    except Exception as e:
        print(f"Error analyzing position with Stockfish: {e}")
        return None

def print_stockfish_analysis(analysis: Dict, shadow_board: chess.Board):
    """
    Print formatted Stockfish analysis results.
    
    Args:
        analysis: The Stockfish analysis result dictionary
        shadow_board: Current chess board state
    """
    if not analysis:
        print("No analysis available.")
        return
    
    eval_score = analysis.get('eval')
    best_move = analysis.get('san')
    win_chance = analysis.get('winChance')
    
    print("\n🧠 Stockfish Analysis:")
    print(f"Evaluation: {eval_score}")
    if eval_score > 0:
        print(f"White has an advantage of {abs(eval_score)} pawns")
    elif eval_score < 0:
        print(f"Black has an advantage of {abs(eval_score)} pawns")
    else:
        print("Position is equal")
    
    if win_chance is not None:
        print(f"Win chance: {win_chance:.2f}%")
    
    if best_move:
        print(f"Best move: {best_move}")
    
    # If a continuation is provided, show the next few moves
    continuation = analysis.get('continuationArr', [])
    if continuation:
        print("\nBest line:")
        try:
            # Create a temporary board from the position
            temp_board = chess.Board(shadow_board.fen())
            line = []
            for uci_move in continuation:
                try:
                    move = chess.Move.from_uci(uci_move)
                    if move in temp_board.legal_moves:
                        line.append(temp_board.san(move))
                        temp_board.push(move)
                except Exception:
                    pass
            
            # Format and print the line
            if line:
                move_num = temp_board.fullmove_number - len(line) // 2
                formatted_line = ""
                for i, san in enumerate(line):
                    if i % 2 == 0:
                        formatted_line += f"{move_num}. {san} "
                        move_num += 1
                    else:
                        formatted_line += f"{san} "
                print(formatted_line)
        except Exception as e:
            print(f"Error formatting continuation: {e}") 