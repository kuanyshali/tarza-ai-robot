import os
import requests
import subprocess

API_KEY = os.environ.get("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

AUDIO_FILE = "speech.wav"

print("Recording... Speak now!")

subprocess.run([
    "arecord",
    "-D", "plughw:2,0",
    "-f", "S16_LE",
    "-r", "16000",
    "-c", "1",
    "-d", "5",
    AUDIO_FILE
], check=True)

print("Recognizing speech...")

with open(AUDIO_FILE, "rb") as audio:
    response = requests.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers={
            "Authorization": "Bearer " + API_KEY
        },
        files={
            "file": ("speech.wav", audio, "audio/wav")
        },
        data={
            "model": "whisper-1",
            "language": "ru"
        },
        timeout=60
    )

response.raise_for_status()

text = response.json()["text"]

print("You said:", text)
