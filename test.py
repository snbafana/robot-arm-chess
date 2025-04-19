"""
quick_cam_vision_v2.py
----------------------
Grab one frame from the chosen webcam, save it, then ask both
OpenAI GPT‑4o and Google Gemini‑2.0‑Flash to analyse the image
with the same prompt.
Tested with:
  • openai>=1.14.0
  • google-generativeai>=0.3.0
  • opencv‑python>=4.10
"""
import cv2, base64, os, sys
from openai import OpenAI
from google import genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ───────────────────────────────
# 0. CONFIGURE API KEYS & CLIENTS
# ───────────────────────────────
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")

if not OPENAI_KEY or not GEMINI_KEY:
    print("Error: Please set both OPENAI_API_KEY and GOOGLE_API_KEY in your .env file")
    sys.exit(1)

openai_client = OpenAI(api_key=OPENAI_KEY)
gemini_client = genai.Client(api_key=GEMINI_KEY)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# ───────────────────────────────
# 1. GLOBAL PROMPT
# ───────────────────────────────
PROMPT = "Describe everything you see and list any obvious safety hazards."

# ───────────────────────────────
# 2. CAMERA HELPERS (Unchanged)
# ───────────────────────────────
def list_available_cameras(max_idx=10):
    cams = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok: cams.append(i)
            cap.release()
    return cams

def tune(cap):
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, -4)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 30)
    cap.set(cv2.CAP_PROP_CONTRAST,   50)
    cap.set(cv2.CAP_PROP_SATURATION, 50)
    cap.set(cv2.CAP_PROP_GAIN,       30)

# ───────────────────────────────
# 3. CAPTURE A FRAME
# ───────────────────────────────
cams = list_available_cameras()
if not cams:
    sys.exit("No camera found.")
cam_idx = cams[1]      # change if you want a different camera
print(f"Using camera {cam_idx}")

cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
tune(cap)
for _ in range(5): cap.read()        # let exposure settle
ok, frame = cap.read()
cap.release()
if not ok: sys.exit("Frame capture failed.")

frame = cv2.convertScaleAbs(frame, alpha=0.9, beta=-5)
cv2.imwrite("capture.jpg", frame)

# ───────────────────────────────
# 4. CALL GPT‑4o VISION
# ───────────────────────────────
base64_image = encode_image("capture.jpg")

oai_resp = openai_client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
)

print("\n── OpenAI GPT‑4o ──")
print(oai_resp.choices[0].message.content.strip())

# ───────────────────────────────
# 5. CALL GEMINI 2.0 FLASH
# ───────────────────────────────
# Upload the image file
myfile = gemini_client.files.upload(file="capture.jpg")

# Generate content using the uploaded file
gemini_resp = gemini_client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[myfile, PROMPT]
)

print("\n── Google Gemini 2.0 Flash ──")
print(gemini_resp.text.strip())
