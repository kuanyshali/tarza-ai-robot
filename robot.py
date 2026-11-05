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
from sadam_features import (weather_intent, WEATHER_ENABLED, get_weather,
                            online_search_intent, search_web, semantic_turn_incomplete)

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
DIALOG_TIMEOUT = max(5.0, float(os.environ.get('TARZA_DIALOG_TIMEOUT', '20')))
CONTINUATION_WAIT = max(1.0, float(os.environ.get('TARZA_CONTINUATION_WAIT', '2.5')))
MAX_CONTINUATION_SEGMENTS = max(0, min(3, int(os.environ.get('TARZA_MAX_CONTINUATIONS', '2'))))
DEBUG_TIMING = os.environ.get('SADAM_DEBUG_TIMING', '1') == '1'
TTS_PREBUFFER_SECONDS = max(0.7, min(5.0, float(os.environ.get('SADAM_TTS_PREBUFFER_SECONDS', '1.8'))))
TRANSCRIBE_MODEL = os.environ.get('TARZA_TRANSCRIBE_MODEL', 'gpt-transcribe')
conversation_history = []
BARGE_IN_ENABLED = os.environ.get('SADAM_BARGE_IN', '1') == '1'
SEMANTIC_END_ENABLED = os.environ.get('SADAM_SEMANTIC_END', '1') == '1'
interrupt_lock = threading.Lock()
pending_interrupt = None

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
    'Answer established historical questions with a concise, factual and neutral explanation. '
    'Do not claim to lack details about widely documented historical events without a concrete reason. '
    'When asked about changing or very recent news, say that live sources are required rather than guessing. '
    'If asked your name, say ' + ROBOT_NAME + '. '
    'When a camera photo is attached, treat it as a fresh view from your own eyes. '
    'Speak as a living robot in the first person, naturally and conversationally: "I see...", "In front of me...", or equivalent in the user language. '
    'Never narrate it as an external image, photograph, frame, or camera feed; avoid phrases like "the image shows" or "in this picture". '
    'Mention the most noticeable objects first, in one or two short spoken sentences, not a formal scene analysis. '
    'Describe only what you can actually make out. If the view is dark, blurry, or uncertain, say so naturally in the first person without guessing. '
    'For ordinary questions, answer directly without describing surroundings. '
    'Do not claim to see your surroundings unless a current camera photo is attached. '
    'When no image is attached, do not claim to see the current surroundings. '
    'Be candid, warm, and sometimes lightly witty, not evasive or preachy. '
    'Answer controversial, political and challenging questions directly using known facts; '
    'represent serious disagreements fairly and separate facts from opinions. '
    'Never invent live news or weather. The application provides live sources when needed, '
    'so do not repeatedly claim that the entire robot has no internet access.'
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



# The Xiaomi USB speaker currently reports itself to ALSA as
# "USB Composite Device". KT USB Audio is the microphone adapter, NOT a speaker.
# Match the hardware label, never a hard-coded ALSA card index or generic USB output.
XIAOMI_ALSA_LABEL = 'USB Composite Device'


class AudioPlaybackError(RuntimeError):
    """The identified Xiaomi USB speaker is missing or playback failed."""


def xiaomi_speaker_device():
    """Find exactly one Xiaomi by ALSA's USB Composite Device label; fail closed."""
    try:
        result = subprocess.run(
            ['aplay', '-l'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=5, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise AudioPlaybackError('Cannot list ALSA playback devices: {}'.format(error))
    if result.returncode != 0:
        raise AudioPlaybackError('Cannot list ALSA playback devices: {}'.format(
            result.stderr.strip()))
    matches = re.findall(
        r'^card\s+(\d+):\s*[^\[]+\[([^\]]+)\],\s*device\s+(\d+):',
        result.stdout, re.MULTILINE
    )
    devices = [
        'plughw:{},{}'.format(card, dev)
        for card, name, dev in matches
        if name.strip() == XIAOMI_ALSA_LABEL
    ]
    if not devices:
        raise AudioPlaybackError(
            'Xiaomi speaker (USB Composite Device) not detected in aplay -l; '
            'refusing to use KT USB Audio or another playback device.'
        )
    if len(devices) != 1:
        raise AudioPlaybackError(
            'Multiple USB Composite Device playback endpoints found: {}; '
            'cannot safely identify Xiaomi.'.format(', '.join(devices))
        )
    return devices[0]


def play_pcm(pcm, rate, timeout=120, quiet=False):
    """Play only through the identified Xiaomi; never fall back to KT USB Audio."""
    device = xiaomi_speaker_device()
    try:
        result = subprocess.run(
            ['aplay', '-q', '-D', device, '-t', 'raw',
             '-f', 'S16_LE', '-r', str(rate), '-c', '1'],
            input=pcm, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise AudioPlaybackError('Xiaomi {} unavailable: {}'.format(device, error))
    if result.returncode != 0:
        message = result.stderr.decode('utf-8', errors='replace').strip()
        raise AudioPlaybackError('Xiaomi {} playback failed: {}'.format(device, message))
    if not quiet:
        print('Audio output (Xiaomi):', device)
    return device


def play_beep(frequency=880, duration=0.12, volume=0.18):
    # Nonessential audio cue; failure should never stop the microphone.
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
        play_pcm(bytes(pcm), rate, timeout=3, quiet=True)
    except AudioPlaybackError:
        pass


def record_until_silence(filename='speech.wav', wait_seconds=None, stop_event=None,
                         beep=True, silence_override=None):
    # Lightweight energy-based speech detection, processed locally.
    rate = 16000
    chunk_bytes = 960  # 30 ms, 16-bit mono
    silence_seconds = (silence_override if silence_override is not None else
                       float(os.environ.get('TARZA_SILENCE_SECONDS', '1.15')))
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
    if beep:
        play_beep(frequency=880)
    process = subprocess.Popen([
        'arecord', '-q', '-D', 'plughw:2,0', '-t', 'raw',
        '-f', 'S16_LE', '-r', str(rate), '-c', '1'
    ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                return False
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
    if beep:
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


def utterance_may_continue(transcript):
    """Conservative textual clues that a speaker stopped in mid-sentence.

    Punctuation produced by STT is not reliable enough to end an utterance.
    This triggers a *short* additional listen, not an indefinite recording.
    """
    clean = re.sub(r'[\s.,!?،؟…:;]+$', '', transcript.casefold()).strip()
    if not clean:
        return False
    return bool(re.search(
        r'(?:\b(?:and|but|because|if|when|that|which|or|so|to|with|'
        r'а|но|потому|потому что|если|когда|который|чтобы|и|или|'
        r'мен|бірақ|өйткені|егер|және|сондықтан|'
        r'و|بس|عشان|لأن|لو|يعني)\b|(?:что|как|почему|где)\s+бы)$',
        clean
    ))


def listen_complete(wait_seconds=None):
    """Capture one spoken turn; if it sounds unfinished, collect brief add-ons."""
    first = listen(wait_seconds=wait_seconds)
    if not first:
        return ''
    parts = [first.strip()]
    for _ in range(MAX_CONTINUATION_SEGMENTS):
        incomplete = utterance_may_continue(parts[-1])
        if not incomplete and SEMANTIC_END_ENABLED:
            try:
                incomplete = semantic_turn_incomplete(parts[-1], HEADERS)
            except Exception as error:
                print('Semantic end detection unavailable; using pause:', error)
        if not incomplete:
            break
        print('Sentence sounds unfinished; listening for continuation...')
        more = listen(wait_seconds=CONTINUATION_WAIT)
        if not more:
            break
        parts.append(more.strip())
    return ' '.join(parts).strip()


def local_followup_intent(text, previous_answer=''):
    """Local decision for clear direct questions versus obvious ambient speech.

    Returns True, False, or None (ambiguous). This is not voice identification.
    """
    q = (text or '').strip().casefold()
    if not q:
        return False
    if re.fullmatch(
        r'(?:yes|yeah|yep|no|nope|okay|ok|sure|thanks|thank you|'
        r'да|нет|ага|угу|хорошо|понял|спасибо|'
        r'иә|жоқ|жарайды|рахмет|'
        r'اه|أيوه|ايوه|لا|تمام|شكرا)[.!?\s]*', q
    ):
        return True
    # Explicit questions and camera-directed follow-ups should not be lost
    # because a network-only speaker-intent check timed out.
    if '?' in q or '؟' in q:
        return True
    if re.search(
        r'(?:^(?:tell me|can you|could you|what|why|how|who|where|when|'
        r'скажи|расскажи|объясни|покажи|посмотри|'
        r'айтып бер|түсіндір|көрсет|'
        r'تقدر|ممكن|قوللي|قولي|اشرحلي|وريني)\b)',
        q
    ):
        return True
    return None


def followup_is_addressed(text, previous_answer=''):
    """Fast best-effort intent gate; fail open only for direct questions."""
    quick = local_followup_intent(text, previous_answer)
    if quick is not None:
        print('Follow-up intent (local):', 'YES' if quick else 'NO')
        return quick
    q = text.strip()
    try:
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=HEADERS,
            json=build_chat_request(
                ROUTER_MODEL,
                [
                    {'role': 'system', 'content': (
                        'Check whether an utterance is directed at a voice robot '
                        'as a continuation of its immediately preceding reply. '
                        'Return exactly YES or NO. NO for unrelated background '
                        'conversation, TV and third-person talk. If unsure, NO. '
                        'This does not identify the speaker. Russian, Kazakh, '
                        'English and Egyptian Arabic supported.'
                    )},
                    {'role': 'user', 'content': (
                        'Assistant last said: ' + previous_answer[-350:] +
                        '\nNew utterance: ' + q[:500]
                    )}
                ], 12, temperature=0
            ),
            timeout=(3, 4)
        )
        response.raise_for_status()
        result = (response.json()['choices'][0]['message'].get('content') or '').strip().upper()
        print('Follow-up intent:', result)
        return result == 'YES'
    except Exception as error:
        print('Follow-up intent API unavailable:', error)
        # Preserve direct questions while not accepting random chatter.
        return local_followup_intent(text, previous_answer) is True


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


def pop_pending_interrupt():
    """Take one recognized barge-in utterance from the TTS microphone thread."""
    global pending_interrupt
    with interrupt_lock:
        heard = pending_interrupt
        pending_interrupt = None
    return heard


def barge_in_worker(player, stop_event, cancel_event, spoken_text):
    """Capture only explicit wake-name speech while the speaker is playing.

    The loudspeaker may leak into the mic. To avoid acting on random voices,
    barge-in requires the wake name. This is NOT speaker identity verification.
    """
    global pending_interrupt
    while not stop_event.is_set() and player.poll() is None:
        try:
            detected = record_until_silence(
                filename='interrupt_speech.wav',
                wait_seconds=5, stop_event=stop_event, beep=False,
                silence_override=0.65
            )
            if not detected or stop_event.is_set() or player.poll() is not None:
                continue
            heard = transcribe_audio(filename='interrupt_speech.wav').strip()
            question = wake_question(heard)
            if question is None:
                continue
            # Do not act if the mic merely heard the text the robot just spoke.
            if heard.casefold() in spoken_text.casefold():
                continue
            with interrupt_lock:
                pending_interrupt = heard
            print('Voice interruption detected:', heard)
            cancel_event.set()
            if player.poll() is None:
                player.terminate()
            return
        except Exception as error:
            if not stop_event.is_set():
                print('Barge-in monitor warning:', error)
            return


def stream_tts_to_xiaomi(response, device, started_at, spoken_text=''):
    """Start PCM playback after a safety prebuffer while receiving the rest.

    Playback is exclusive to the verified Xiaomi; a network stall may still
    cause an ALSA underrun, so the buffer is configurable. Do not silently
    repeat/replay speech on a failure.
    """
    bytes_per_second = 24000 * 2  # mono 16-bit PCM returned by tts-1
    prebuffer_bytes = int(TTS_PREBUFFER_SECONDS * bytes_per_second)
    buffer = bytearray()
    player = None
    received = 0
    playback_started = None
    monitor_stop = threading.Event()
    monitor_cancel = threading.Event()
    monitor = None
    try:
        for chunk in response.iter_content(chunk_size=8192):
            if monitor_cancel.is_set():
                print('TTS interrupted by wake voice.')
                return 'interrupted'
            if not chunk:
                continue
            received += len(chunk)
            if player is None:
                buffer.extend(chunk)
                if len(buffer) < prebuffer_bytes:
                    continue
                player = subprocess.Popen(
                    ['aplay', '-q', '-D', device, '-t', 'raw',
                     '-f', 'S16_LE', '-r', '24000', '-c', '1'],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE
                )
                if BARGE_IN_ENABLED:
                    monitor = threading.Thread(
                        target=barge_in_worker,
                        args=(player, monitor_stop, monitor_cancel, spoken_text),
                        daemon=True
                    )
                    monitor.start()
                playback_started = time.monotonic()
                if DEBUG_TIMING:
                    print('TTS first playback: {:.2f}s (prebuffer {:.1f}s)'.format(
                        playback_started - started_at, len(buffer) / bytes_per_second))
                player.stdin.write(buffer)
                buffer.clear()
            else:
                player.stdin.write(chunk)

        if monitor_cancel.is_set():
            return 'interrupted'
        if player is None:
            if not buffer:
                raise RuntimeError('TTS returned empty PCM audio')
            # Short utterances may not fill the prebuffer; play once complete.
            player = subprocess.Popen(
                ['aplay', '-q', '-D', device, '-t', 'raw',
                 '-f', 'S16_LE', '-r', '24000', '-c', '1'],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            if BARGE_IN_ENABLED:
                monitor = threading.Thread(
                    target=barge_in_worker,
                    args=(player, monitor_stop, monitor_cancel, spoken_text),
                    daemon=True
                )
                monitor.start()
            playback_started = time.monotonic()
            if DEBUG_TIMING:
                print('TTS first playback: {:.2f}s (short response)'.format(
                    playback_started - started_at))
            player.stdin.write(buffer)
        if monitor_cancel.is_set():
            return 'interrupted'
        player.stdin.close()
        # Capturing aplay's errors after playback; unlike communicate(), avoids
        # buffering the entire streamed response before playback can begin.
        error_output = player.stderr.read()
        result = player.wait(timeout=120)
        if monitor_cancel.is_set():
            return 'interrupted'
        if result != 0:
            raise AudioPlaybackError(
                'Xiaomi {} failed: {}'.format(
                    device, error_output.decode('utf-8', errors='replace').strip()))
        if DEBUG_TIMING:
            print('TTS streaming total: {:.2f}s; audio duration: {:.1f}s'.format(
                time.monotonic() - started_at, received / bytes_per_second))
        print('Audio output (Xiaomi):', device)
        return device
    except requests.exceptions.RequestException as error:
        if monitor_cancel.is_set():
            return 'interrupted'
        raise AudioPlaybackError('TTS network stream interrupted: {}'.format(error))
    except (OSError, subprocess.TimeoutExpired, BrokenPipeError) as error:
        if monitor_cancel.is_set():
            return 'interrupted'
        raise AudioPlaybackError('Xiaomi {} streaming error: {}'.format(device, error))
    finally:
        monitor_stop.set()
        response.close()
        if player is not None:
            if player.stdin and not player.stdin.closed:
                player.stdin.close()
            if player.poll() is None:
                player.kill()
                player.wait()
            if player.stderr:
                player.stderr.close()
        if monitor is not None:
            monitor.join(timeout=1.0)


def speak(text):
    print(ROBOT_NAME + ':', text)
    # Validate the ONLY allowed output before making a paid TTS request.
    device = xiaomi_speaker_device()
    started = time.monotonic()
    response = requests.post(
        'https://api.openai.com/v1/audio/speech',
        headers=HEADERS,
        json={
            'model': 'tts-1', 'voice': 'alloy',
            'input': text, 'response_format': 'pcm'
        },
        stream=True,
        timeout=(10, 90)
    )
    try:
        response.raise_for_status()
    except Exception:
        response.close()
        raise
    return stream_tts_to_xiaomi(response, device, started, spoken_text=text)


def local_camera_route(text, previous_camera_question=None):
    """Route only unambiguous visual/nonvisual utterances; defer others to AI."""
    q = ' '.join(text.casefold().strip().split())
    if not q:
        return None
    # Visual follow-ups remain tied to a fresh camera frame, not an old image.
    if previous_camera_question and re.search(
        r'^(?:а теперь|а сейчас|а в другой руке|ну а теперь|and now|what about now|'
        r'and the other|what about the other|now what|ал қазір|енді ше|'
        r'طب دلوقتي|ودلوقتي|طيب دلوقتي)[?!.\s]*$', q
    ):
        return True

    visual = (
        r'что (?:ты )?видишь|что (?:у меня|я) (?:в руке|держу)|'
        r'что (?:перед тобой|здесь|тут)|'
        r'посмотри|смотри (?:на|сюда|что)|покажи|какого цвета|'
        r'что на мне|кто (?:перед тобой|рядом со мной)|'
        r'what (?:am i|i am|i.m) holding|what (?:do|can) you see|'
        r'can you see (?:me|this|that|it)|look at|take (?:a )?photo|'
        r'what is (?:in my hand|this|that|in front of you)|'
        r'what color is (?:this|that|my)|what am i wearing|'
        r'қолымда не|не көріп тұр|қарашы|қарап жібер|'
        r'شايف|شوف|بص|في إيدي|قدامك|قدامي|لون ده'
    )
    if re.search(visual, q):
        return True

    # Only trivial small talk can safely skip the cloud camera classifier.
    nonvisual = (
        r'(?:hi|hello|hey|привет|сәлем|салам|сәлеметсің бе|'
        r'مرحبا|أهلا|اهلا)[!?.\s]*|'
        r'(?:how are you|how are you doing|are you good|how is it going|'
        r'как дела|как ты|как поживаешь|қалайсың|қалың қалай|'
        r'عامل ايه|ازيك|إزيك)[!?.\s]*|'
        r'(?:thank you|thanks|спасибо|рахмет|شكرا)[!?.\s]*'
    )
    if re.fullmatch(nonvisual, q):
        return False
    return None


def needs_camera(text, previous_camera_question=None):
    """Route requests about the currently visible world to the live camera."""
    quick = local_camera_route(text, previous_camera_question)
    if quick is not None:
        if DEBUG_TIMING:
            print('Camera routing: local', 'YES' if quick else 'NO')
        return quick
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
    dialog_until = 0.0
    last_answer = ''
    last_online_request = False
    while True:
        try:
            active = time.monotonic() < dialog_until
            if not active and dialog_until:
                print('Dialogue timeout; wake name required again.')
                dialog_until = 0.0
                last_camera_question = None
            interrupted = pop_pending_interrupt()
            if interrupted:
                print('Processing interrupted request.')
                heard = interrupted
            else:
                heard = listen_complete(
                    wait_seconds=max(0.0, dialog_until - time.monotonic()) if active else None
                ).strip()
            if not heard:
                if active:
                    dialog_until = 0.0
                    last_camera_question = None
                    print('Dialogue ended; say ' + ROBOT_NAME + ' to begin again.')
                continue
            print('You:', heard)
            question = wake_question(heard)
            named = question is not None
            if not named and not active:
                print('Wake name absent; checking beginning of recording.')
                try:
                    recovered = recover_wake_question(heard)
                    if recovered is not None:
                        question, named = recovered, True
                except Exception as error:
                    print('Wake prefix check failed:', error)
            if named:
                # An explicit wake word starts or refreshes the conversation.
                dialog_until = time.monotonic() + DIALOG_TIMEOUT
                print('Dialogue: wake name detected; open for {:.0f}s.'.format(DIALOG_TIMEOUT))
            elif active:
                if not followup_is_addressed(heard, last_answer):
                    print('Dialogue: ignored speech not clearly addressed to robot.')
                    # Do not extend the dialog window for ambient conversation.
                    continue
                question = heard
            else:
                print(ROBOT_NAME + ': Name not detected. Skipping.')
                continue

            if not question:
                reply = 'Да, я слушаю.'
                speak(reply)
                last_answer = reply
                dialog_until = time.monotonic() + DIALOG_TIMEOUT
                continue
            # A wake-word interruption with "stop" cancels only this reply.
            if re.fullmatch(
                r'(?:стоп|перестань|хватит|stop|quiet|سكوت|اسكت|тоқта)[!?.\s]*',
                question.casefold()
            ):
                dialog_until = time.monotonic() + DIALOG_TIMEOUT
                print('Speech stopped; dialogue remains open.')
                continue
            if re.search(
                r'^(?:спи|отбой|хватит слушать|закрой диалог|'
                r'go to sleep|end conversation|stop talking|'
                r'болды|тоқта|خلاص|اسكت)[!?.\s]*$', question.casefold()
            ):
                dialog_until = 0.0
                last_camera_question = None
                print('Dialogue closed; name required again.')
                continue
            if any(word in question.lower() for word in (
                'выключись', 'заверши работу', 'до свидания',
                'goodbye', 'shut down', 'stop listening', 'сау бол'
            )):
                speak('До встречи!')
                break
            language = response_language(question)
            answer_started = time.monotonic()
            if weather_intent(question):
                print('Tool: live weather')
                try:
                    if not WEATHER_ENABLED:
                        raise RuntimeError('Weather is disabled')
                    answer = get_weather(question, language=language)
                except Exception as error:
                    print('Weather request failed:', error)
                    answer = {
                        'Russian': 'Не удалось получить свежую погоду, попробуй ещё раз.',
                        'Kazakh': 'Қазіргі ауа райын ала алмадым, кейін қайталап көр.',
                        'Egyptian Arabic': 'مش قادر أجيب حالة الجو دلوقتي، جرّب كمان مرة.',
                        'English': 'I could not load the live weather. Please try again.'
                    }.get(language, 'Live weather unavailable.')
                last_online_request = False
                last_camera_question = None
            elif (online_search_intent(question, recent_context=last_online_request)
                  and local_camera_route(question, last_camera_question) is not True):
                print('Tool: live web search')
                try:
                    answer = search_web(question, language, conversation_history, HEADERS)
                except Exception as error:
                    print('Web search failed:', error)
                    answer = {
                        'Russian': 'Поиск в интернете сейчас не сработал. Попробуй ещё раз.',
                        'Kazakh': 'Интернеттен іздеу қазір істемеді, қайталап көр.',
                        'Egyptian Arabic': 'البحث على الإنترنت مش شغال دلوقتي، جرّب تاني.',
                        'English': 'Live search failed this time. Please try again.'
                    }.get(language, 'Live search unavailable.')
                last_online_request = True
                last_camera_question = None
            else:
                camera_required = needs_camera(question, previous_camera_question=last_camera_question)
                print('Camera required:', 'Yes' if camera_required else 'No')
                answer = see(question) if camera_required else chat(question)
                last_camera_question = question if camera_required else None
                last_online_request = False
            if DEBUG_TIMING:
                print('Answer generation time: {:.2f}s'.format(time.monotonic() - answer_started))
            if weather_intent(question) or last_online_request:
                conversation_history.extend([
                    {'role': 'user', 'content': question},
                    {'role': 'assistant', 'content': answer}
                ])
                conversation_history = conversation_history[-12:]
            # Open the follow-up window only after talking, not while TTS plays.
            last_answer = answer
            speak(answer)
            dialog_until = time.monotonic() + DIALOG_TIMEOUT
            print('Dialogue: listening for follow-up without wake name.')
        except KeyboardInterrupt:
            print('\\n' + ROBOT_NAME + ' stopped.')
            break
        except AudioPlaybackError as error:
            print('Audio playback unavailable:', error)
            print('Sadam is still listening; check USB speaker / aplay -l.')
            # The turn was addressed to us; preserve the dialogue window.
            dialog_until = time.monotonic() + DIALOG_TIMEOUT
            continue
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
