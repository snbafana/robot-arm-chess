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
from dotenv import load_dotenv
import os
from google import genai
from fish_audio_sdk import WebSocketSession, TTSRequest
from pydub import AudioSegment
import simpleaudio
from io import BytesIO
import random

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

    load_dotenv()

    google_api_key = os.getenv('GEMINI_KEY')
    fish_api_key = os.getenv('FISH_KEY')

    def queryAI(prompt):
        print("Querying Gemini")
        client = genai.Client(api_key=google_api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        return response.text

    def play_saved_audio(file_path):
        """
        Play an MP3 file using pydub and simpleaudio.
        """
        try:
            # Verify and load the MP3 file
            audio = AudioSegment.from_mp3(file_path)
            # Export to raw PCM (WAV format) in memory for playback
            pcm_buffer = BytesIO()
            audio.export(pcm_buffer, format="wav")
            pcm_buffer.seek(0)
            
            # Play PCM audio using simpleaudio
            wav_audio = simpleaudio.WaveObject.from_wave_file(pcm_buffer)
            play_obj = wav_audio.play()
            play_obj.wait_done()  # Wait until playback is complete
            print("Playback completed.")
        except Exception as e:
            print(f"Playback error: {e}")

    def speakMessage(message):
        sync_websocket = WebSocketSession(fish_api_key)

        if not message:
            message = "I would normally trashtalk you . but I ran out of AI credits ."

        def stream():
            for line in message.split():
                yield line + " "

        tts_request = TTSRequest(
            text="",  # Empty text for streaming
            reference_id="76bb6ae7b26c41fbbd484514fdb014c2",  # VOICE CHOSEN
            temperature=0.7,  # Controls randomness in speech generation
            top_p=0.7,  # Controls diversity via nucleus sampling
        )

        try:
            # Save audio chunks to file
            output_file = "output.mp3"
            with open(output_file, "wb") as f:
                print("Generating and saving audio...")
                chunk_count = 0
                for chunk in sync_websocket.tts(
                    tts_request,
                    stream(),  # Use streaming
                    backend="speech-1.6"  # Specify which TTS model to use
                ):
                    f.write(chunk)
                    chunk_count += 1
                    print(f"Received chunk {chunk_count}: {len(chunk)} bytes")

            # Verify file exists and has data
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                print(f"Error: {output_file} was not created or is empty.")
                return

            print(f"Saved audio to {output_file} ({os.path.getsize(output_file)} bytes)")
            
            # Play the saved audio
            play_saved_audio(output_file)

        except Exception as e:
            print(f"TTS error: {e}")

    def trashtalk(current_lead):
        if current_lead > 60:
            trash = queryAI(f"""You are a great chess player, far above your opponent. 
            You are very confident, make up a single sentence, clever, full of yourself, 
            poke at your opponent about your odds of winning. 
            Your elite mental model assumes you are currently leading by a weighted score of {current_lead}, 
            don't disclose this, but make your quip reflectedly confident.
            MAKE IT BRIEF MAKE IT BRIEF
            ONLY USE WORDS, DO NOT USE SYMBOLS OTHER THAN NORMAL PUNCTUATION""")
        elif current_lead > 40:
            trash = queryAI(f"""You are a great chess player, far above your opponent. 
            But you are losing.
            Make up a quip to knock your opponent off guard so that you can regain your lead. 
            Your professional mental model assumes your odds of winning are currently {current_lead}/100, 
            don't let them know you know this, but make your quip reflectedly confident and aggressive. 
            MAKE IT BRIEF MAKE IT BRIEF
            ONLY USE WORDS, DO NOT USE SYMBOLS OTHER THAN NORMAL PUNCTUATION""")
        else:
            filler_phrases = [
            "Hmm, interesting.",
            "Let me think about this.",
            "That's tricky.",
            "I didn't expect that move.",
            "Okay, okay...",
            "What's the best way here?",
            "Alright, let's see.",
            "There's got to be something better.",
            "Hold on a second.",
            "If I do that, then...",
            "Maybe... no, wait.",
            "This position is tense.",
            "One wrong move...",
            "Let's not rush this.",
            "How bad could it be?"]
            trash = random.choice(filler_phrases)

        speakMessage(trash)
    try: trashtalk(70)
    except: print("something went wrong trying to trashtalk, double check API keys")
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