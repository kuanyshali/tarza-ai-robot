import os
import requests
import subprocess

API_KEY = os.environ.get("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

def speak(text):
    print("Robot:", text)

    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={
            "Authorization": "Bearer " + API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "model": "tts-1",
            "voice": "alloy",
            "input": text,
            "response_format": "mp3"
        },
        timeout=60
    )

    response.raise_for_status()

    with open("response.mp3", "wb") as f:
        f.write(response.content)

    subprocess.run([
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", "response.mp3",
        "-ar", "48000",
        "-ac", "2",
        "response.wav"
    ], check=True)

    subprocess.run([
        "aplay",
        "-D", "plughw:3,0",
        "response.wav"
    ], check=True)


if __name__ == "__main__":
    speak(
        "Привет! Я твой искусственный интеллект. "
        "Я умею слышать тебя, видеть окружающий мир "
        "и разговаривать с тобой."
    )
