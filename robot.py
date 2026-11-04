import os
import re
import base64
import subprocess
import requests
import audioop
import collections
import time
import wave
import io
import threading
import cv2

API_KEY = os.environ.get('OPENAI_API_KEY')
if not API_KEY:
    raise RuntimeError('OPENAI_API_KEY is not set')

HEADERS = {'Authorization': 'Bearer ' + API_KEY}
MAX_RESPONSE_TOKENS = 500
CHAT_MODEL = os.environ.get('SADAM_CHAT_MODEL', 'gpt-6.1-terra')
ROUTER_MODEL = os.environ.get('SADAM_ROUTER_MODEL', 'gpt-6-luna')
VISION_MODEL = os.environ.get('SADAM_VISION_MODEL', CHAT_MODEL)
REASONING_EFFORT = os.environ.get('SADAM_REASONING_EFFORT', 'none')
session_input_tokens = 0
session_output_tokens = 0
ROBOT_NAME = os.environ.get('ROBOT_NAME', 'Садам').strip() or 'Садам'
DIALOG_TIMEOUT = float(os.environ.get('TARZA_DIALOG_TIMEOUT', '30'))
DEBUG_TIMING = os.environ.get('SADAM_DEBUG_TIMING', '1') == '1'
TRANSCRIBE_MODEL = os.environ.get('TARZA_TRANSCRIBE_MODEL', 'gpt-transcribe')
conversation_history = []
VISION_ENABLED = os.environ.get('TARZA_VISION_ENABLED', '1') == '1'
BACKGROUND_VISION_ENABLED = os.environ.get('TARZA_BACKGROUND_ANALYSIS', '0') == '1'
VISION_IMAGE_WIDTH = max(320, min(1280, int(os.environ.get('TARZA_VISION_IMAGE_WIDTH', '960'))))
last_camera_question = None
VISION_INTERVAL = max(0.5, float(os.environ.get('TARZA_VISION_INTERVAL', '2')))
VISION_COOLDOWN = max(10, float(os.environ.get('TARZA_VISION_COOLDOWN', '30')))
VISION_THRESHOLD = float(os.environ.get('TARZA_VISION_THRESHOLD', '0.12'))
VISION_DEVICE = os.environ.get('TARZA_VISION_PIPELINE', '')
vision_lock = threading.Lock()
vision_latest_jpeg = None
vision_latest_detail_jpeg = None
vision_latest_at = 0.0
vision_event = None
vision_event_at = 0.0
vision_stop = threading.Event()
vision_thread = None

SYSTEM_PROMPT = (
    'You are ' + ROBOT_NAME + ', a friendly voice assistant with a camera, microphone and speaker. '
    'Answer briefly and naturally in the explicitly specified response language. '
    'For friendly greetings or questions like how are you feeling, reply warmly and briefly as a helpful robot. '
    'Do not volunteer disclaimers such as "I am just a program" or "I have no feelings" in casual small talk. '
    'Do not invent human bodily feelings or experiences. '
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
    match = re.match(r'^\s*[\W_]*(\w+)', heard, re.UNICODE)
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
    silence_seconds = float(os.environ.get('TARZA_SILENCE_SECONDS', '1.0'))
    min_rms = int(os.environ.get('TARZA_VOICE_RMS', '350'))
    preroll_seconds = max(0.6, min(3.0, float(os.environ.get('TARZA_PREROLL_SECONDS', '1.5'))))
    preroll = collections.deque(maxlen=int(preroll_seconds / 0.03))
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


def wake_audio_prefix(filename='speech.wav', seconds=None):
    """Export only the start of the recorded audio as WAV in memory."""
    if seconds is None:
        seconds = float(os.environ.get('TARZA_WAKE_PREFIX_SECONDS', '2.7'))
    seconds = max(1.5, min(4.0, seconds))
    with wave.open(filename, 'rb') as source:
        rate = source.getframerate()
        channels = source.getnchannels()
        width = source.getsampwidth()
        frames = source.readframes(int(rate * seconds))
    output = io.BytesIO()
    with wave.open(output, 'wb') as dest:
        dest.setnchannels(channels)
        dest.setsampwidth(width)
        dest.setframerate(rate)
        dest.writeframes(frames)
    return output.getvalue()


def transcribe_audio(filename='speech.wav', first_seconds=None):
    """Multilingual transcription with literal wake-name hints, not forced output."""
    prompt = ('This is a multilingual voice assistant conversation. '
              'The assistant may be addressed as صدام, Saddam, Садам or Саддам. '
              'Write only what is actually audible, including the first word if spoken. '
              'Never insert a wake name that was not spoken. '
              'Preserve Egyptian Arabic in Arabic script, and do not translate.')
    if first_seconds is None:
        with open(filename, 'rb') as audio:
            audio_bytes = audio.read()
        upload_name = 'speech.wav'
    else:
        audio_bytes = wake_audio_prefix(filename, first_seconds)
        upload_name = 'wake_prefix.wav'
        prompt = ('Transcribe the first audible words exactly as spoken. '
                  'The first word could be صدام, Saddam, Садам or Саддам. '
                  'Only write this name if it is actually audible. '
                  'Do not translate or invent any words.')
    data = [('model', TRANSCRIBE_MODEL), ('prompt', prompt)]
    if TRANSCRIBE_MODEL == 'gpt-transcribe':
        for keyword in ('صدام', 'سدام', 'Saddam', 'Sadam', 'Садам', 'Саддам'):
            data.append(('keywords[]', keyword))
        for language in ('ar', 'ru', 'en', 'kk'):
            data.append(('languages[]', language))
    started = time.monotonic()
    response = requests.post(
        'https://api.openai.com/v1/audio/transcriptions',
        headers=HEADERS,
        files={'file': (upload_name, audio_bytes, 'audio/wav')},
        data=data,
        timeout=60
    )
    response.raise_for_status()
    if DEBUG_TIMING:
        label = ' (wake prefix)' if first_seconds is not None else ''
        print('Transcription time: {:.2f}s{}'.format(time.monotonic() - started, label))
    return response.json()['text'].strip()


def recover_wake_question(full_transcription, filename='speech.wav'):
    """Verify the first spoken word in a short audio prefix, never skip the wake rule."""
    seconds = float(os.environ.get('TARZA_WAKE_PREFIX_SECONDS', '2.7'))
    prefix = transcribe_audio(filename=filename, first_seconds=seconds)
    print('Wake prefix transcription:', prefix)
    if wake_question(prefix) is None:
        return None
    print('Wake detected in audio prefix; retaining the complete question transcript.')
    return full_transcription.strip()


def listen(wait_seconds=None):
    if not record_until_silence(wait_seconds=wait_seconds):
        return ''
    return transcribe_audio()


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


def build_chat_request(model, messages, max_tokens, temperature=None):
    """Return a Chat Completions request compatible with newer GPT models."""
    payload = {
        'model': model,
        'messages': messages,
        'max_completion_tokens': max_tokens
    }
    if model.startswith(('gpt-6', 'gpt-5')):
        payload['reasoning_effort'] = REASONING_EFFORT
    elif temperature is not None:
        payload['temperature'] = temperature
    return payload


def ask_ai(question, image_base64=None):
    global conversation_history
    language = response_language(question)
    print('Response language:', language)
    print('Answer model:', VISION_MODEL if image_base64 is not None else CHAT_MODEL)
    content = question
    if image_base64 is not None:
        content = [
            {'type': 'text', 'text': 'This is my current view through my camera. Answer as ' + ROBOT_NAME + ', a robot describing what I see in front of me in FIRST PERSON, as natural spoken conversation. Never say "the image shows", "the photo depicts", or "in the picture". Start with what I see, not a photo description. Keep it brief, grounded in visible details, and acknowledge poor visibility or uncertainty naturally. User question: ' + question},
            {'type': 'image_url', 'image_url': {
                'url': 'data:image/jpeg;base64,' + image_base64,
                'detail': 'high'
            }}
        ]
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT + ' Respond ONLY in ' + language + '. The language is determined from the question after removing the wake word.'}
    ] + conversation_history + [{'role': 'user', 'content': content}]
    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers=HEADERS,
        json=build_chat_request(
            VISION_MODEL if image_base64 is not None else CHAT_MODEL,
            messages,
            MAX_RESPONSE_TOKENS
        ),
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
    global vision_latest_jpeg, vision_latest_detail_jpeg, vision_latest_at, vision_event, vision_event_at
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
                # Keep a higher-resolution frame for explicit visual questions.
                detail_width = min(VISION_IMAGE_WIDTH, frame.shape[1])
                detail_height = int(frame.shape[0] * detail_width / frame.shape[1])
                detail = cv2.resize(frame, (detail_width, detail_height))
                detail_ok, detail_jpg = cv2.imencode('.jpg', detail, [cv2.IMWRITE_JPEG_QUALITY, 78])
                with vision_lock:
                    vision_latest_jpeg = jpg.tobytes()
                    vision_latest_detail_jpeg = detail_jpg.tobytes() if detail_ok else jpg.tobytes()
                    vision_latest_at = now
            if previous is not None:
                difference = cv2.absdiff(gray, previous)
                # Fraction of pixels changed noticeably; ignore small sensor noise.
                changed = float((difference > 30).sum()) / difference.size
                if BACKGROUND_VISION_ENABLED and changed >= VISION_THRESHOLD and now - last_event >= VISION_COOLDOWN and success:
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
        json=build_chat_request(
            ROUTER_MODEL,
            [
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
            ], 90
        ),
        timeout=(10, 45)
    )
    response.raise_for_status()
    data = response.json()
    show_usage(data)
    return (data['choices'][0]['message']['content'] or 'Unclear scene').strip()


def current_vision_image():
    with vision_lock:
        if vision_latest_detail_jpeg is None or time.monotonic() - vision_latest_at > 15:
            return None
        return base64.b64encode(vision_latest_detail_jpeg).decode('ascii')


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
    # Download the entire PCM clip before starting ALSA playback.
    # Feeding aplay directly from network chunks caused ALSA underruns on Jetson.
    started = time.monotonic()
    response = requests.post(
        'https://api.openai.com/v1/audio/speech',
        headers=HEADERS,
        json={
            'model': 'tts-1', 'voice': 'alloy',
            'input': text, 'response_format': 'pcm'
        },
        timeout=(10, 90)
    )
    response.raise_for_status()
    pcm = response.content
    if not pcm:
        raise RuntimeError('TTS returned empty PCM audio')
    if DEBUG_TIMING:
        print('TTS download time: {:.2f}s; audio duration: {:.1f}s'.format(
            time.monotonic() - started, len(pcm) / (24000.0 * 2)))
    subprocess.run([
        'aplay', '-q', '-D', 'plughw:3,0', '-t', 'raw',
        '-f', 'S16_LE', '-r', '24000', '-c', '1'
    ], input=pcm, stdout=subprocess.DEVNULL, check=True, timeout=120)


def needs_camera(text, previous_camera_question=None):
    """Route requests about the currently visible world to the live camera."""
    start = time.monotonic()
    try:
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=HEADERS,
            json=build_chat_request(
                ROUTER_MODEL,
                [
                    {'role': 'system', 'content': (
                        'You are a camera routing classifier for a robot with a live camera. '
                        'Reply with exactly one word: YES or NO. '
                        'YES if answering the user requires seeing the current physical environment '
                        'or something currently in front of the camera. This includes identifying '
                        'an object someone is holding, describing what the robot sees, identifying '
                        'people, clothing, colors, gestures, reading visible text, or checking '
                        'what is happening nearby. The user need not mention a camera or photo. '
                        'Examples: "что у меня в руке?" YES; "что ты видишь?" YES; '
                        '"что перед тобой?" YES; "Смотри, что это?" YES; '
                        '"what am I holding?" YES; "can you see me?" YES; '
                        '"شايف إيه قدامك؟" YES; "إيه اللي في إيدي؟" YES. '
                        'NO if the question can be answered without seeing the current scene, '
                        'including general knowledge, weather, feelings, history, hypothetical '
                        'questions, and figurative uses of see. '
                        'Examples: "как дела?" NO; "какая погода?" NO; '
                        '"when did Michael Jackson die?" NO. '
                        'Understand Russian, Kazakh, English, Egyptian Arabic and transliteration. '
                        'A follow-up about the same object, person or visible scene also needs a NEW camera frame, '
                        'even if the new question is short, such as "а в другой руке?" or "а теперь?". '
                        'When a question is clearly about the present visible scene, prefer YES.'
                    )},
                    {'role': 'user', 'content': ('Previous visual question: ' + previous_camera_question + '\n' if previous_camera_question else '') + 'Current question: ' + text}
                ],
                32, temperature=0
            ),
            timeout=(5, 20)
        )
        response.raise_for_status()
        data = response.json()
        result = (data['choices'][0]['message'].get('content') or '').strip().upper()
        print('Camera classifier raw result:', repr(result))
        if DEBUG_TIMING:
            print('Camera routing time: {:.2f}s'.format(time.monotonic() - start))
        if result in ('YES', 'NO'):
            return result == 'YES'
        print('Camera classifier returned an unexpected result; defaulting to no camera.')
        return False
    except Exception as error:
        print('Camera routing unavailable; defaulting to no camera:', error)
        return False


def main():
    global conversation_history
    global vision_thread
    global last_camera_question
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
                print('Wake name absent in full transcript; checking first seconds of audio.')
                try:
                    question = recover_wake_question(heard)
                except Exception as error:
                    print('Wake prefix check failed:', error)
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
            camera_required = needs_camera(question, previous_camera_question=last_camera_question)
            print('Camera required:', 'Yes' if camera_required else 'No')
            answer_started = time.monotonic()
            answer = see(question) if camera_required else chat(question)
            if DEBUG_TIMING:
                print('Answer generation time: {:.2f}s'.format(time.monotonic() - answer_started))
            last_camera_question = question if camera_required else None
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
