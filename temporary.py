# Dependencies/Packages Required:
# - python-dotenv: For loading environment variables from .env file
#   Install: pip install python-dotenv
# - fish_audio_sdk: For Fish Audio TTS functionality
#   Install: pip install fish_audio_sdk
# - pydub: For handling MP3 audio files
#   Install: pip install pydub
# - simpleaudio: For playing audio
#   Install: pip install simpleaudio
# - ffmpeg: Required by pydub for MP3 decoding
#   Install:
#     Windows: Download from ffmpeg.org and add to PATH
#     Mac: brew install ffmpeg
#     Linux: sudo apt-get install ffmpeg (Ubuntu) or equivalent

from dotenv import load_dotenv
import os
from google import genai
from fish_audio_sdk import WebSocketSession, TTSRequest
from pydub import AudioSegment
import simpleaudio
from io import BytesIO
import random

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

# Example usage
if __name__ == "__main__":
    # print(queryAI("Explain how AI works in ten words"))
    trashtalk(70)