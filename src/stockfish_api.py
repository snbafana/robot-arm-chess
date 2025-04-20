"""
stockfish_api.py
---------------
Stockfish chess engine API integration.
Provides functions for analyzing chess positions using Stockfish via WebSocket.
"""
import asyncio
import websockets
import json
import chess
from typing import Dict, Optional, List, Union

class ChessAPIClient:
    def __init__(self, url='wss://chess-api.com/v1'):
        self.url = url
        self.websocket = None

    async def connect(self):
        """Establish a WebSocket connection to the chess API."""
        self.websocket = await websockets.connect(self.url)
        return self.websocket is not None

    async def analyze_position(self, fen: str, depth: int = 12, 
                              variants: int = 1, max_thinking_time: int = 50) -> Optional[Dict]:
        """
        Send a FEN position for analysis and receive responses.
        
        Args:
            fen: FEN notation of the position
            depth: Analysis depth (1-18)
            variants: Number of variants to analyze (max 5)
            max_thinking_time: Maximum thinking time in ms (max 100)
            
        Returns:
            Dictionary with analysis results or None if failed
        """
        try:
            if not self.websocket:
                await self.connect()
            
            # Prepare the request
            request = {
                "fen": fen,
                "variants": min(5, variants),
                "depth": min(18, depth),
                "maxThinkingTime": min(100, max_thinking_time)
            }
            
            # Send the request
            await self.websocket.send(json.dumps(request))
            
            # Wait for response(s)
            best_move = None
            
            # Listen for responses
            while True:
                try:
                    response = await self.websocket.recv()
                    data = json.loads(response)
                    
                    # If this is a "bestmove" response, use it and break
                    if data.get("type") == "bestmove":
                        best_move = data
                        break
                    
                    # Otherwise keep the latest response
                    best_move = data
                    
                    # If we've reached the desired depth, break
                    if data.get("depth", 0) >= depth:
                        break
                        
                except websockets.exceptions.ConnectionClosed:
                    break
            
            return best_move
        
        except Exception as e:
            print(f"Error analyzing position with Stockfish: {e}")
            return None
    
    async def close(self):
        """Close the WebSocket connection."""
        if self.websocket:
            await self.websocket.close()

async def analyze_position_with_stockfish_async(fen: str, depth: int = 12) -> Optional[Dict]:
    """
    Analyze a chess position using the Stockfish API asynchronously.
    
    Args:
        fen: FEN notation of the position
        depth: Analysis depth (1-18)
        
    Returns:
        Dictionary with analysis results or None if failed
    """
    client = ChessAPIClient()
    try:
        await client.connect()
        result = await client.analyze_position(fen=fen, depth=depth)
        return result
    finally:
        await client.close()

def analyze_position_with_stockfish(fen: str, depth: int = 12) -> Optional[Dict]:
    """
    Analyze a chess position using the Stockfish API.
    Synchronous wrapper around the async function.
    
    Args:
        fen: FEN notation of the position
        depth: Analysis depth (1-18)
        
    Returns:
        Dictionary with analysis results or None if failed
    """
    return asyncio.run(analyze_position_with_stockfish_async(fen, depth))

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
    best_move = analysis.get('text') or analysis.get('san')
    win_chance = analysis.get('winChance')
    
    print("\n🧠 Stockfish Analysis:")
    print(f"Evaluation: {eval_score}")
    if eval_score and eval_score > 0:
        print(f"White has an advantage of {abs(eval_score)} pawns")
    elif eval_score and eval_score < 0:
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