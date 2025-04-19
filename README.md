# Chess Vision Analyzer

This project uses computer vision and AI to analyze chess moves by comparing board states before and after each move.

## Setup

1. Install UV (Python package manager):
```bash
pip install uv
```

2. Create a virtual environment and install dependencies:
```bash
uv venv
uv pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
```

## Usage

1. Position your camera above the chess board
2. Run the script:
```bash
uv run chess_vision.py
```

3. Follow the prompts:
   - The script will capture the initial board setup
   - After each opponent's move, press Enter
   - The script will capture the new board state
   - Analysis from both OpenAI and Gemini will be displayed
   - This process repeats until you stop the script

## Output

- Images are saved in the `captures/` directory
- Move history is saved in `captures/moves.json`
- Each move includes:
  - Initial and current board images
  - Analysis from both AI models
  - Timestamp of the move

## Requirements

- Python 3.8+
- Webcam
- OpenAI API key
- Google API key
- UV package manager
