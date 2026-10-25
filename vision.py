import subprocess

print("Capturing new photo...")

subprocess.run(
    ["python3", "camera.py"],
    check=True
)
import os
import base64
import requests

API_KEY = os.environ.get("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY first")

with open("photo.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "model": "gpt-4o-mini",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Опиши на русском языке, что ты видишь "
                        "на изображении. Отвечай кратко и естественно."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + image_base64,
    "detail": "low"
                    }
                }
            ]
        }
    ],
    "max_tokens": 300
}

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={
        "Authorization": "Bearer " + API_KEY,
        "Content-Type": "application/json"
    },
    json=payload,
    timeout=60
)

if response.status_code != 200:
    print("HTTP status:", response.status_code)
    print("API error:", response.text)
    raise SystemExit(1)

result = response.json()

print("\nAI:", result["choices"][0]["message"]["content"])

usage = result["usage"]

print("\n--- API Usage ---")
print("Input tokens:", usage["prompt_tokens"])
print("Output tokens:", usage["completion_tokens"])
print("Total tokens:", usage["total_tokens"])

cost = (
    usage["prompt_tokens"] * 0.15 / 1_000_000
    + usage["completion_tokens"] * 0.60 / 1_000_000
)

print("Estimated cost: ${:.6f}".format(cost))
