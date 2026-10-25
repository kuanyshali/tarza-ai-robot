import os
import re
import base64
import subprocess
import requests
import audioop
import collections
import time
import wave

API_KEY = os.environ.get('OPENAI_API_KEY')
if not API_KEY:
    raise RuntimeError('OPENAI_API_KEY is not set')

HEADERS = {'Authorization': 'Bearer ' + API_KEY}
MAX_RESPONSE_TOKENS = 500
session_input_tokens = 0
session_output_tokens = 0

SYSTEM_PROMPT = (
    'You are Tarza, a friendly voice assistant with a camera, microphone and speaker. '
    'Answer briefly and naturally in the explicitly specified response language. '
    'Do not translate English questions into Russian. '
    'If asked your name, say Tarza. '
    'When an image is attached, it is a current photo captured by your own camera. '
    'You can inspect that photo and should answer questions about what is visible directly and naturally, as Tarza. '
    'Never say you cannot see images or your surroundings when a photo is attached. '
    'Describe only what the attached photo supports; if something is unclear, say so. '
    'When no image is attached, do not claim to see the current surroundings.'
)

WAKE_WORDS = ('tarza', 'тарза', 'тарзан', 'тарса')
WAKE_PATTERN = re.compile(r'(?<!\w)(?:' + '|'.join(map(re.escape, WAKE_WORDS)) + r')(?!\w)', re.IGNORECASE)


def show_usage(data):
    global session_input_tokens, session_output_tokens
    usage = data.get('usage', {})
    input_tokens = usage.get('prompt_tokens', 0)
    output_tokens = usage.get('completion_tokens', 0)
    session_input_tokens += input_tokens
    session_output_tokens += output_tokens
    print('\nTARZA - CHAT TOKEN USAGE')
    print('Input tokens:       {}'.format(input_tokens))
    print('Output tokens:      {}'.format(output_tokens))
    print('Total tokens:       {}'.format(input_tokens + output_tokens))
    print('Response allowance: {} / {}'.format(max(0, MAX_RESPONSE_TOKENS - output_tokens), MAX_RESPONSE_TOKENS))
    print('Session chat total: {}'.format(session_input_tokens + session_output_tokens))
    print('(Whisper and TTS usage is not included)')


def record_until_silence(filename='speech.wav'):
    # Lightweight energy-based speech detection, processed locally.
    rate = 16000
    chunk_bytes = 960  # 30 ms, 16-bit mono
    silence_seconds = float(os.environ.get('TARZA_SILENCE_SECONDS', '1.5'))
    min_rms = int(os.environ.get('TARZA_VOICE_RMS', '450'))
    preroll = collections.deque(maxlen=10)
    frames = []
    speaking = False
    last_voice = None
    started_at = None
    noise_floor = 0.0

    print('Listening... Speak naturally; recording ends after a pause.')
    process = subprocess.Popen([
        'arecord', '-q', '-D', 'plughw:2,0', '-t', 'raw',
        '-f', 'S16_LE', '-r', str(rate), '-c', '1'
    ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            chunk = process.stdout.read(chunk_bytes)
            if len(chunk) != chunk_bytes:
                raise RuntimeError('Microphone recording stopped unexpectedly')
            rms = audioop.rms(chunk, 2)
            now = time.monotonic()
            if not speaking:
                noise_floor = (noise_floor * 0.97 + rms * 0.03) if noise_floor else rms
                threshold = max(min_rms, int(noise_floor * 2.5))
                preroll.append(chunk)
                if rms >= threshold:
                    speaking = True
                    started_at = now
                    last_voice = now
                    frames.extend(preroll)
                    print('Speech detected...')
            else:
                frames.append(chunk)
                threshold = max(min_rms, int(noise_floor * 2.0))
                if rms >= threshold:
                    last_voice = now
                if now - last_voice >= silence_seconds:
                    break
                if now - started_at >= 45:
                    print('Maximum recording length reached.')
                    break
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    with wave.open(filename, 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b''.join(frames))


def listen():
    record_until_silence()
    with open('speech.wav', 'rb') as audio:
        response = requests.post(
            'https://api.openai.com/v1/audio/transcriptions',
            headers=HEADERS,
            files={'file': ('speech.wav', audio, 'audio/wav')},
            data={
                'model': 'gpt-4o-mini-transcribe',
                'prompt': 'Transcribe the spoken words verbatim in their original language. Never translate. Keep English speech in English, Russian speech in Russian, and Kazakh speech in Kazakh. The assistant name is Tarza.'
            },
            timeout=60
        )
    response.raise_for_status()
    return response.json()['text']


def response_language(question):
    # The wake word is already removed. Prefer English for mixed Latin/Cyrillic questions.
    if re.search(r'[a-zA-Z]', question):
        return 'English'
    if re.search(r'[әғқңөұүһіӘҒҚҢӨҰҮҺІ]', question):
        return 'Kazakh'
    if re.search(r'[а-яА-ЯёЁ]', question):
        return 'Russian'
    return 'English'


def ask_ai(question, image_base64=None):
    language = response_language(question)
    print('Response language:', language)
    content = question
    if image_base64 is not None:
        content = [
            {'type': 'text', 'text': 'The following image is a photo just captured by my camera. Answer the user question using what is actually visible in the image. Do not disclaim image or vision capabilities. User question: ' + question},
            {'type': 'image_url', 'image_url': {
                'url': 'data:image/jpeg;base64,' + image_base64,
                'detail': 'low'
            }}
        ]
    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers=HEADERS,
        json={
            'model': 'gpt-4o-mini',
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT + ' Respond ONLY in ' + language + '. The language is determined from the question after removing the wake word.'},
                {'role': 'user', 'content': content}
            ],
            'max_completion_tokens': MAX_RESPONSE_TOKENS
        },
        timeout=90
    )
    response.raise_for_status()
    data = response.json()
    show_usage(data)
    answer = data['choices'][0]['message']['content']
    return answer or 'Unable to generate a response.'


def see(question):
    print('Capturing photo...')
    subprocess.run(['python3', 'camera.py'], check=True)
    with open('photo.jpg', 'rb') as image:
        encoded = base64.b64encode(image.read()).decode('utf-8')
    return ask_ai(question, image_base64=encoded)


def chat(question):
    return ask_ai(question)


def speak(text):
    print('Tarza:', text)
    # Stream 24 kHz, 16-bit mono PCM directly to the existing USB speaker.
    # This avoids waiting for the full MP3 and avoids the FFmpeg conversion.
    with requests.post(
        'https://api.openai.com/v1/audio/speech',
        headers=HEADERS,
        json={
            'model': 'tts-1', 'voice': 'alloy',
            'input': text, 'response_format': 'pcm'
        },
        stream=True,
        timeout=(10, 90)
    ) as response:
        response.raise_for_status()
        player = subprocess.Popen([
            'aplay', '-q', '-D', 'plughw:3,0', '-t', 'raw',
            '-f', 'S16_LE', '-r', '24000', '-c', '1'
        ], stdin=subprocess.PIPE)
        try:
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    player.stdin.write(chunk)
            player.stdin.close()
            if player.wait() != 0:
                raise RuntimeError('Audio playback failed')
        finally:
            if player.poll() is None:
                player.kill()
                player.wait()


def needs_camera(text):
    text = text.lower()
    # Recognize English variations that do not match exact phrases.
    if re.search(r'\b(see|seeing|look|looking|watch|watching)\b', text):
        if any(word in text for word in ('you', 'your', 'front', 'around', 'right now', 'camera', 'at this', 'at that')):
            return True
    keywords = [
        'видишь', 'видно', 'посмотри', 'перед тобой', 'перед собой',
        'камера', 'покажи', 'что это', 'какой предмет', 'какого цвета',
        'what do you see', 'what can you see', 'what are you seeing', 'what do you see in front of you', 'look at',
        'take a picture', 'take a photo', 'what is this', "what's this",
        'what color', 'what colour', 'describe what you see',
        'не көріп тұрсың', 'не көріп отырсың', 'қарашы',
        'мынау не', 'түсі қандай', 'алдыңда не'
    ]
    return any(keyword in text for keyword in keywords)


def main():
    print('Tarza started. Press Ctrl+C to exit.')
    while True:
        try:
            heard = listen().strip()
            print('You:', heard)
            if not heard:
                continue
            if not WAKE_PATTERN.search(heard):
                print('Tarza: Wake word not detected. Skipping.')
                continue
            question = WAKE_PATTERN.sub('', heard).strip(' ,.!?')
            if not question:
                speak('Да, я слушаю.')
                continue
            if any(word in question.lower() for word in (
                'выключись', 'заверши работу', 'до свидания',
                'goodbye', 'shut down', 'stop listening', 'сау бол'
            )):
                speak('До встречи!')
                break
            camera_required = needs_camera(question)
            print('Camera required:', 'Yes' if camera_required else 'No')
            answer = see(question) if camera_required else chat(question)
            speak(answer)
        except KeyboardInterrupt:
            print('\nTarza stopped.')
            break
        except Exception as error:
            print('Error:', error)
            print('Tarza stopped. Check the microphone and audio device.')
            break

if __name__ == '__main__':
    main()
