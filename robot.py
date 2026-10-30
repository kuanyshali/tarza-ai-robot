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
ROBOT_NAME = os.environ.get('ROBOT_NAME', 'Tarza').strip() or 'Tarza'
DIALOG_TIMEOUT = float(os.environ.get('TARZA_DIALOG_TIMEOUT', '30'))
conversation_history = []

SYSTEM_PROMPT = (
    'You are ' + ROBOT_NAME + ', a friendly voice assistant with a camera, microphone and speaker. '
    'Answer briefly and naturally in the explicitly specified response language. '
    'Do not translate English questions into Russian. '
    'If asked your name, say ' + ROBOT_NAME + '. '
    'When a camera photo is attached, treat it as a fresh view from your own eyes. '
    'Speak as a living robot in the first person, naturally and conversationally: "I see...", "In front of me...", or equivalent in the user language. '
    'Never narrate it as an external image, photograph, frame, or camera feed; avoid phrases like "the image shows" or "in this picture". '
    'Mention the most noticeable objects first, in one or two short spoken sentences, not a formal scene analysis. '
    'Describe only what you can actually make out. If the view is dark, blurry, or uncertain, say so naturally in the first person without guessing. '
    'Do not claim to see your surroundings unless a current camera photo is attached. '
    'When no image is attached, do not claim to see the current surroundings.'
)

_wake_words_env = os.environ.get('ROBOT_WAKE_WORDS', '').strip()
if _wake_words_env:
    WAKE_WORDS = tuple(w.strip() for w in _wake_words_env.split(',') if w.strip())
elif ROBOT_NAME.lower() == 'tarza':
    WAKE_WORDS = ('tarza', 'тарза', 'тарзан', 'тарса')
else:
    WAKE_WORDS = (ROBOT_NAME,)
if not WAKE_WORDS:
    WAKE_WORDS = (ROBOT_NAME,)
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



def play_beep(frequency=880, duration=0.12, volume=0.18):
    # Play a short local cue through the USB speaker before/after recording.
    import math
    import struct

    rate = 48000
    count = int(rate * duration)
    pcm = bytearray()
    for i in range(count):
        fade = max(0.0, min(1.0, i / 240.0, (count - i) / 240.0))
        value = int(32767 * volume * fade * math.sin(2 * math.pi * frequency * i / rate))
        pcm.extend(struct.pack('<h', value))
    try:
        subprocess.run(
            ['aplay', '-q', '-D', 'plughw:3,0', '-t', 'raw',
             '-f', 'S16_LE', '-r', str(rate), '-c', '1'],
            input=bytes(pcm), stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=3, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def record_until_silence(filename='speech.wav', wait_seconds=None):
    # Lightweight energy-based speech detection, processed locally.
    rate = 16000
    chunk_bytes = 960  # 30 ms, 16-bit mono
    silence_seconds = float(os.environ.get('TARZA_SILENCE_SECONDS', '1.5'))
    min_rms = int(os.environ.get('TARZA_VOICE_RMS', '350'))
    preroll = collections.deque(maxlen=10)
    frames = []
    speaking = False
    last_voice = None
    started_at = None
    noise_floor = 0.0
    wait_started = time.monotonic()

    print('Listening... Speak naturally; recording ends after a pause.')
    play_beep(frequency=880)
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
                if wait_seconds is not None and now - wait_started >= wait_seconds:
                    return False
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
    print('Recording finished. Processing...')
    play_beep(frequency=600, duration=0.16)
    return True


def listen(wait_seconds=None):
    if not record_until_silence(wait_seconds=wait_seconds):
        return ''
    with open('speech.wav', 'rb') as audio:
        response = requests.post(
            'https://api.openai.com/v1/audio/transcriptions',
            headers=HEADERS,
            files={'file': ('speech.wav', audio, 'audio/wav')},
            data={
                'model': 'gpt-4o-mini-transcribe',
                'prompt': 'Transcribe the spoken words verbatim in their original language. Never translate. Keep English speech in English, Russian speech in Russian, and Kazakh speech in Kazakh. The assistant name is ' + ROBOT_NAME + '.'
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
    global conversation_history
    language = response_language(question)
    print('Response language:', language)
    content = question
    if image_base64 is not None:
        content = [
            {'type': 'text', 'text': 'This is my current view through my camera. Answer as Tarza, a robot describing what I see in front of me in FIRST PERSON, as natural spoken conversation. Never say "the image shows", "the photo depicts", or "in the picture". Start with what I see, not a photo description. Keep it brief, grounded in visible details, and acknowledge poor visibility or uncertainty naturally. User question: ' + question},
            {'type': 'image_url', 'image_url': {
                'url': 'data:image/jpeg;base64,' + image_base64,
                'detail': 'low'
            }}
        ]
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT + ' Respond ONLY in ' + language + '. The language is determined from the question after removing the wake word.'}
    ] + conversation_history + [{'role': 'user', 'content': content}]
    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers=HEADERS,
        json={
            'model': 'gpt-4o-mini',
            'messages': messages,
            'max_completion_tokens': MAX_RESPONSE_TOKENS
        },
        timeout=90
    )
    response.raise_for_status()
    data = response.json()
    show_usage(data)
    answer = data['choices'][0]['message']['content']
    answer = answer or 'Unable to generate a response.'
    # Keep only text in history, never large base64 images.
    conversation_history.extend([
        {'role': 'user', 'content': question + (' [A camera photo was provided.]' if image_base64 is not None else '')},
        {'role': 'assistant', 'content': answer}
    ])
    conversation_history = conversation_history[-12:]
    return answer


def see(question):
    print('Capturing photo...')
    subprocess.run(['python3', 'camera.py'], check=True)
    with open('photo.jpg', 'rb') as image:
        encoded = base64.b64encode(image.read()).decode('utf-8')
    return ask_ai(question, image_base64=encoded)


def chat(question):
    return ask_ai(question)


def speak(text):
    print(ROBOT_NAME + ':', text)
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
    global conversation_history
    print(ROBOT_NAME + ' started. Press Ctrl+C to exit.')
    active_until = 0.0
    while True:
        try:
            remaining = max(0.0, active_until - time.monotonic())
            if active_until and remaining == 0:
                active_until = 0.0
                conversation_history.clear()
                print('Conversation ended. Say ' + ROBOT_NAME + ' to start again.')
            heard = listen(wait_seconds=remaining if active_until else None).strip()
            if not heard:
                if active_until:
                    active_until = 0.0
                    conversation_history.clear()
                    print('Conversation timed out. Say ' + ROBOT_NAME + ' to start again.')
                continue
            print('You:', heard)
            woke = bool(WAKE_PATTERN.search(heard))
            if not active_until and not woke:
                print(ROBOT_NAME + ': Wake word not detected. Skipping.')
                continue
            if woke and not active_until:
                conversation_history.clear()
            question = WAKE_PATTERN.sub('', heard).strip(' ,.!?') if woke else heard.strip(' ,.!?')
            if not question:
                speak('Да, я слушаю.')
                active_until = time.monotonic() + DIALOG_TIMEOUT
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
            active_until = time.monotonic() + DIALOG_TIMEOUT
        except KeyboardInterrupt:
            print('\\n' + ROBOT_NAME + ' stopped.')
            break
        except Exception as error:
            print('Error:', error)
            print(ROBOT_NAME + ' stopped. Check the microphone and audio device.')
            break

if __name__ == '__main__':
    main()
