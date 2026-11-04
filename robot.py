import os
import re
import base64
import subprocess
import requests
import audioop
import collections
import time
import wave
import threading
import cv2

API_KEY = os.environ.get('OPENAI_API_KEY')
if not API_KEY:
    raise RuntimeError('OPENAI_API_KEY is not set')

HEADERS = {'Authorization': 'Bearer ' + API_KEY}
MAX_RESPONSE_TOKENS = 500
session_input_tokens = 0
session_output_tokens = 0
ROBOT_NAME = os.environ.get('ROBOT_NAME', 'Садам').strip() or 'Садам'
DIALOG_TIMEOUT = float(os.environ.get('TARZA_DIALOG_TIMEOUT', '30'))
conversation_history = []
VISION_ENABLED = os.environ.get('TARZA_VISION_ENABLED', '1') == '1'
VISION_INTERVAL = max(0.5, float(os.environ.get('TARZA_VISION_INTERVAL', '2')))
VISION_COOLDOWN = max(10, float(os.environ.get('TARZA_VISION_COOLDOWN', '30')))
VISION_THRESHOLD = float(os.environ.get('TARZA_VISION_THRESHOLD', '0.12'))
VISION_DEVICE = os.environ.get('TARZA_VISION_PIPELINE', '')
vision_lock = threading.Lock()
vision_latest_jpeg = None
vision_latest_at = 0.0
vision_event = None
vision_event_at = 0.0
vision_stop = threading.Event()
vision_thread = None

SYSTEM_PROMPT = (
    'You are ' + ROBOT_NAME + ', a friendly voice assistant with a camera, microphone and speaker. '
    'Answer briefly and naturally in the explicitly specified response language. '
    'Do not translate English questions into Russian. '
    'When the response language is Egyptian Arabic, use natural everyday Egyptian Masri dialect, not formal Modern Standard Arabic, and write in Arabic script. '
    'Understand Egyptian Arabic and Modern Standard Arabic input. '
    'If asked your name, say ' + ROBOT_NAME + '. '
    'When a camera photo is attached, treat it as a fresh view from your own eyes. '
    'Speak as a living robot in the first person, naturally and conversationally: "I see...", "In front of me...", or equivalent in the user language. '
    'Never narrate it as an external image, photograph, frame, or camera feed; avoid phrases like "the image shows" or "in this picture". '
    'Mention the most noticeable objects first, in one or two short spoken sentences, not a formal scene analysis. '
    'Describe only what you can actually make out. If the view is dark, blurry, or uncertain, say so naturally in the first person without guessing. '
    'For ordinary questions, answer directly without describing surroundings. '
    'Do not claim to see your surroundings unless a current camera photo is attached. '
    'When no image is attached, do not claim to see the current surroundings.'
)

_wake_words_env = os.environ.get('ROBOT_WAKE_WORDS', '').strip()
if _wake_words_env:
    WAKE_WORDS = tuple(w.strip() for w in _wake_words_env.split(',') if w.strip())
elif ROBOT_NAME.casefold() in ('садам', 'sadam', 'саддам', 'saddam'):
    # Short transcription variants for Russian, Kazakh and English accents.
    WAKE_WORDS = (
        'садам', 'саддам', 'садамм', 'саддан', 'садан', 'сада',
        'садэм', 'саддэм', 'сэдэм', 'сэддам', 'сэдем',
        'адам', 'аддам', 'адэм', 'адамм',
        'sadam', 'saddam', 'sadamm', 'saddamm', 'sadan',
        'sadem', 'saddem', 'sedam', 'sedem', 'sada',
        'صدام', 'سدام', 'صدامم', 'سدامم',
        'adam', 'addam', 'adem'
    )
else:
    WAKE_WORDS = (ROBOT_NAME,)
if not WAKE_WORDS:
    WAKE_WORDS = (ROBOT_NAME,)
WAKE_PATTERN = re.compile(r'(?<!\w)(?:' + '|'.join(map(re.escape, sorted(set(WAKE_WORDS), key=len, reverse=True))) + r')(?!\w)', re.IGNORECASE)
WAKE_CANDIDATES = tuple(set(WAKE_WORDS) | {'садом', 'саднам', 'سعدتم'})
ARABIC_LATIN_HINTS = re.compile(r'\b(?:enta|inta|enti|ezayak|izzayak|shayef|shaif|shif|eih|delwaqt|dilwaqt|ana|mesh|mish|fein|fen|bet|bt)\b', re.I)

def wake_question(heard):
    # Wake only when the first word is a name or a close transcription.
    match = re.match(r'^\s*[\W_]*([\w\u0600-\u06FF]+)', heard, re.UNICODE)
    if not match:
        return None
    word = match.group(1).casefold()
    if word not in (w.casefold() for w in WAKE_CANDIDATES):
        from difflib import SequenceMatcher
        cyr = bool(re.search(r'[а-яё]', word))
        arab = bool(re.search(r'[\u0600-\u06FF]', word))
        latin = bool(re.search(r'[a-z]', word))
        if not (cyr or arab or latin) or len(word) < 4 or len(word) > 8:
            return None
        candidates = [w.casefold() for w in WAKE_CANDIDATES
                      if (bool(re.search(r'[а-яё]', w)) == cyr
                          and bool(re.search(r'[\u0600-\u06FF]', w)) == arab
                          and bool(re.search(r'[a-z]', w)) == latin)]
        similarity = max((SequenceMatcher(None, word, w).ratio() for w in candidates), default=0)
        if similarity < 0.78:
            return None
        print('Wake name approximation:', word, 'score:', round(similarity, 2))
    return heard[match.end():].strip(' ,.!?،؟')


def show_usage(data):
    global session_input_tokens, session_output_tokens
    usage = data.get('usage', {})
    input_tokens = usage.get('prompt_tokens', 0)
    output_tokens = usage.get('completion_tokens', 0)
    session_input_tokens += input_tokens
    session_output_tokens += output_tokens
    print('\n' + ROBOT_NAME + ' - CHAT TOKEN USAGE')
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
                'prompt': 'Transcribe the spoken words verbatim in their original language. Never translate. Keep English speech in English, Russian speech in Russian, Kazakh speech in Kazakh, and Arabic speech (including Egyptian dialect) in Arabic. Accented name variants include Садом, Саднам, سعدتم. The assistant name can be صدام or سدام. The assistant name is ' + ROBOT_NAME + '.'
            },
            timeout=60
        )
    response.raise_for_status()
    return response.json()['text']


def response_language(question):
    if re.search(r'[\u0600-\u06FF]', question):
        return 'Egyptian Arabic'
    if len(ARABIC_LATIN_HINTS.findall(question)) >= 2:
        return 'Egyptian Arabic'
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
            {'type': 'text', 'text': 'This is my current view through my camera. Answer as ' + ROBOT_NAME + ', a robot describing what I see in front of me in FIRST PERSON, as natural spoken conversation. Never say "the image shows", "the photo depicts", or "in the picture". Start with what I see, not a photo description. Keep it brief, grounded in visible details, and acknowledge poor visibility or uncertainty naturally. User question: ' + question},
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



def camera_pipeline():
    return VISION_DEVICE or (
        'nvarguscamerasrc ! video/x-raw(memory:NVMM), '
        'width=1280, height=720, framerate=30/1 ! '
        'nvvidconv ! video/x-raw, format=BGRx ! '
        'videoconvert ! video/x-raw, format=BGR ! '
        'appsink drop=true max-buffers=1 sync=false'
    )


def vision_worker():
    """Single camera owner: local motion detection; cloud calls only for events."""
    global vision_latest_jpeg, vision_latest_at, vision_event, vision_event_at
    camera = None
    previous = None
    last_event = 0.0
    failures = 0
    try:
        camera = cv2.VideoCapture(camera_pipeline(), cv2.CAP_GSTREAMER)
        if not camera.isOpened():
            print('Vision: camera unavailable; voice remains active.')
            return
        print('Vision: continuous camera and local change detection started.')
        while not vision_stop.is_set():
            ok, frame = camera.read()
            if not ok:
                failures += 1
                if failures >= 10:
                    print('Vision: camera read failed repeatedly; stopping vision.')
                    return
                vision_stop.wait(0.3)
                continue
            failures = 0
            now = time.monotonic()
            # Downscale for low CPU load and for inexpensive low-detail vision calls.
            small = cv2.resize(frame, (320, 180))
            gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (11, 11), 0)
            success, jpg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if success:
                with vision_lock:
                    vision_latest_jpeg = jpg.tobytes()
                    vision_latest_at = now
            if previous is not None:
                difference = cv2.absdiff(gray, previous)
                # Fraction of pixels changed noticeably; ignore small sensor noise.
                changed = float((difference > 30).sum()) / difference.size
                if changed >= VISION_THRESHOLD and now - last_event >= VISION_COOLDOWN and success:
                    last_event = now
                    print('Vision: significant change ({:.0%}); analyzing event...'.format(changed))
                    try:
                        description = analyze_vision_event(jpg.tobytes())
                        with vision_lock:
                            vision_event = description
                            vision_event_at = time.monotonic()
                        print('Vision event:', description)
                    except Exception as error:
                        print('Vision: cloud analysis failed:', error)
            previous = gray
            vision_stop.wait(VISION_INTERVAL)
    except Exception as error:
        print('Vision: worker error:', error)
    finally:
        if camera is not None:
            camera.release()


def analyze_vision_event(jpeg):
    """Bounded, silent cloud event description. No automatic speech."""
    image = base64.b64encode(jpeg).decode('ascii')
    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers=HEADERS,
        json={
            'model': 'gpt-4o-mini',
            'messages': [
                {'role': 'system', 'content': (
                    'You are a visual event observer. Describe only clearly visible '
                    'objects or actions in one short factual sentence. Do not invent '
                    'movement or claim what changed from a single frame. '
                    'Do not identify people. If unclear, say "Unclear scene".')},
                {'role': 'user', 'content': [
                    {'type': 'text', 'text': 'Describe the current scene briefly.'},
                    {'type': 'image_url', 'image_url': {
                        'url': 'data:image/jpeg;base64,' + image, 'detail': 'low'}}
                ]}
            ],
            'max_completion_tokens': 90
        },
        timeout=(10, 45)
    )
    response.raise_for_status()
    data = response.json()
    show_usage(data)
    return (data['choices'][0]['message']['content'] or 'Unclear scene').strip()


def current_vision_image():
    with vision_lock:
        if vision_latest_jpeg is None or time.monotonic() - vision_latest_at > 15:
            return None
        return base64.b64encode(vision_latest_jpeg).decode('ascii')


def current_vision_event():
    with vision_lock:
        if vision_event is None or time.monotonic() - vision_event_at > 120:
            return None
        return vision_event


def see(question):
    encoded = current_vision_image()
    if encoded is None:
        if VISION_ENABLED:
            return ask_ai(question + ' (Camera is temporarily unavailable; do not claim to see.)')
        print('Capturing photo...')
        subprocess.run(['python3', 'camera.py'], check=True)
        with open('photo.jpg', 'rb') as image:
            encoded = base64.b64encode(image.read()).decode('utf-8')
    return ask_ai(question, image_base64=encoded)


def chat(question):
    # Do not inject background camera observations into ordinary conversation.
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
    """Use a small semantic classifier instead of fixed camera keywords."""
    try:
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=HEADERS,
            json={
                'model': 'gpt-4o-mini',
                'temperature': 0,
                'max_completion_tokens': 12,
                'messages': [
                    {'role': 'system', 'content': (
                        'Decide if the user requests information from the robot LIVE CAMERA. '
                        'Return only YES or NO. YES for seeing or inspecting the current surroundings, '
                        'visible people, objects, appearance, colors, actions, reading visible text. '
                        'NO for general knowledge, weather, feelings, past events, hypotheticals, '
                        'and figurative uses of seeing. Understand Russian, English, Kazakh, '
                        'Egyptian Arabic, Modern Standard Arabic and transliterated Arabic. '
                        'When uncertain, answer NO.')},
                    {'role': 'user', 'content': text}
                ]
            },
            timeout=(5, 15)
        )
        response.raise_for_status()
        result = (response.json()['choices'][0]['message']['content'] or '').strip().upper()
        return result == 'YES'
    except Exception as error:
        print('Camera routing unavailable; defaulting to no camera:', error)
        return False


def main():
    global conversation_history
    global vision_thread
    print(ROBOT_NAME + ' started. Press Ctrl+C to exit.')
    if VISION_ENABLED:
        vision_thread = threading.Thread(target=vision_worker, daemon=True)
        vision_thread.start()
    while True:
        try:
            heard = listen(wait_seconds=None).strip()
            if not heard:
                continue
            print('You:', heard)
            question = wake_question(heard)
            if question is None:
                print(ROBOT_NAME + ': Name not detected. Skipping.')
                continue
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
            print('\\n' + ROBOT_NAME + ' stopped.')
            break
        except Exception as error:
            print('Error:', error)
            print(ROBOT_NAME + ' stopped. Check the microphone and audio device.')
            break

if __name__ == '__main__':
    try:
        main()
    finally:
        vision_stop.set()
        if vision_thread is not None:
            vision_thread.join(timeout=3)
