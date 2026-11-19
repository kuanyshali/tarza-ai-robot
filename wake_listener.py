"""Offline, opt-in Vosk wake-word listener for Jetson Nano.

Runs no transcription or OpenAI calls until the local recognizer spots the
wake word. A single uninterrupted audio segment is saved, so the subsequent
cloud transcription still sees "Aisana, <question>" as one utterance.
"""
import audioop
import collections
import json
import os
import re
import subprocess
import time
import wave

RATE = 16000
CHUNK = 3200  # 100 ms of PCM16 mono
WAKE_WORDS = (
    'айсана', 'эйсана', 'айсанаа', 'aisana', 'aysana',
    'آيسانا', 'ايسانا', 'أيسانا'
)
# Strict, word-boundary-matched local activation. No fuzzy matching:
# incidental background conversations must not wake the robot.
WORDS = set(WAKE_WORDS)


def has_wake_word(transcript):
    """Only the first spoken word activates Aisana."""
    text = (transcript or '').strip().casefold()
    tokens = re.findall(r'[^\W_]+', text, flags=re.UNICODE)
    return bool(tokens and tokens[0] in WORDS)


def extract_recognition_text(result):
    """Decode Vosk final/partial result; tolerate empty and malformed JSON."""
    try:
        data = json.loads(result)
        return (data.get('text') or data.get('partial') or '').strip()
    except (ValueError, TypeError, AttributeError):
        return ''


def record_wake_phrase(model, filename='speech.wav', mic_device='plughw:2,0',
                       min_rms=350, silence_seconds=1.15,
                       max_seconds=45, stop_event=None):
    """Block until locally recognized wake word, then capture the full utterance.

    The model is instantiated once in robot.main and reused. A single arecord
    process owns the mic throughout detection, avoiding card-busy conflicts.
    The saved WAV includes audio from before the trigger (preroll), allowing
    cloud STT to transcribe the complete wake+question phrase.
    """
    from vosk import KaldiRecognizer
    recognizer = KaldiRecognizer(model, RATE)
    preroll = collections.deque(maxlen=16)  # last 1.6 seconds
    utterance = []
    active = False
    trigger = False
    last_loud = None
    start = None
    noise = 0.0
    print('Wake listener: offline, waiting for "Aisana"...')
    process = subprocess.Popen(
        ['arecord', '-q', '-D', mic_device, '-t', 'raw',
         '-f', 'S16_LE', '-r', str(RATE), '-c', '1'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                return False
            chunk = process.stdout.read(CHUNK)
            if len(chunk) != CHUNK:
                raise RuntimeError('Wake microphone stopped unexpectedly')
            level = audioop.rms(chunk, 2)
            now = time.monotonic()
            preroll.append(chunk)
            if not active:
                noise = (0.97 * noise + 0.03 * level) if noise else level
                if level >= max(min_rms, int(noise * 2.5)):
                    active = True
                    last_loud = now
                    start = now
                    utterance = list(preroll)
                    recognizer.Reset()
                    # Recognition starts with the complete preroll.
                    for frame in utterance:
                        recognizer.AcceptWaveform(frame)
                    candidate = extract_recognition_text(recognizer.PartialResult())
                    trigger = has_wake_word(candidate)
                continue
            utterance.append(chunk)
            if recognizer.AcceptWaveform(chunk):
                candidate = extract_recognition_text(recognizer.Result())
            else:
                candidate = extract_recognition_text(recognizer.PartialResult())
            if not trigger and has_wake_word(candidate):
                trigger = True
                print('Wake listener: recognized name locally.')
            if level >= max(min_rms, int(noise * 2.0)):
                last_loud = now
            elapsed = now - start
            ended = now - last_loud >= silence_seconds or elapsed >= max_seconds
            if not ended:
                continue
            if not trigger:
                trigger = has_wake_word(extract_recognition_text(recognizer.FinalResult()))
            if trigger:
                with wave.open(filename, 'wb') as dest:
                    dest.setnchannels(1)
                    dest.setsampwidth(2)
                    dest.setframerate(RATE)
                    dest.writeframes(b''.join(utterance))
                print('Wake listener: phrase saved for multilingual transcription.')
                return True
            # Ignore unrelated conversations locally, without any API request.
            active = False
            trigger = False
            utterance = []
            preroll.clear()
            recognizer.Reset()
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def load_vosk_model(path):
    """Validate optional offline dependency and model location explicitly."""
    if not os.path.isdir(path):
        raise RuntimeError(
            'Offline wake model missing at {}. Install it first or set '
            'SADAM_WAKE_MODE=legacy.'.format(path)
        )
    try:
        from vosk import Model, SetLogLevel
    except ImportError:
        raise RuntimeError(
            'Python package vosk is not installed. Run python3 -m pip install vosk '
            'or set SADAM_WAKE_MODE=legacy.'
        )
    SetLogLevel(-1)
    return Model(path)


# -- realtime mode: spot the name in the live microphone stream ----------------
# Vosk's small Russian model does not know the word "Айсана". With a small
# grammar it reliably hears it as the two words "ай" + "сана" with confidence 1.0;
# the other entries soak up similar sounds ("а сам" -> "а сана", "айфон" -> "ай"
# with lower confidence). Tested with synthetic speech: 6 of 6 calls found, 1
# false hit in 10 other phrases ("купи мне айфон"). A false hit costs little:
# the cloud transcript must still contain the name before she answers.
SPOTTER_GRAMMAR = ['ай сана', 'а сана', 'оксана', 'асана', '[unk]']
SPOTTER_CONFIDENCE = float(os.environ.get('AISANA_LOCAL_WAKE_CONFIDENCE', '0.95'))


def name_in_words(words, confidence=SPOTTER_CONFIDENCE):
    """Vosk word list [{'word', 'conf', 'start', 'end'}] -> True if "ай сана" was said."""
    for first, second in zip(words, words[1:]):
        if (first.get('word') == 'ай' and second.get('word') == 'сана'
                and first.get('conf', 0) >= confidence and second.get('conf', 0) >= confidence
                and second.get('start', 0) - first.get('end', 0) < 0.35):
            return True
    return False


class WakeSpotter(object):
    """Feed 24 kHz PCM16 mono chunks; feed() returns True when the name is heard."""

    def __init__(self, model, rate=24000, recognizer=None):
        self.rate = rate
        self.state = None
        if recognizer is None:
            from vosk import KaldiRecognizer
            recognizer = KaldiRecognizer(model, RATE, json.dumps(SPOTTER_GRAMMAR, ensure_ascii=False))
            recognizer.SetWords(True)
        self.recognizer = recognizer

    def feed(self, pcm):
        if self.rate != RATE:
            pcm, self.state = audioop.ratecv(pcm, 2, 1, self.rate, RATE, self.state)
        if not self.recognizer.AcceptWaveform(pcm):
            return False                  # Vosk gives word confidences only at a pause
        try:
            words = json.loads(self.recognizer.Result()).get('result') or []
        except (ValueError, TypeError, AttributeError):
            return False
        return name_in_words(words)


def spotter_from_environment():
    """WakeSpotter when AISANA_LOCAL_WAKE=1 and Vosk + model are installed, else None."""
    if os.environ.get('AISANA_LOCAL_WAKE', '0') != '1':
        return None
    path = os.path.expanduser(os.environ.get(
        'AISANA_VOSK_MODEL', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          'models', 'vosk-model-small-ru-0.22')))
    try:
        spotter = WakeSpotter(load_vosk_model(path))
    except Exception as exc:
        print('Local wake word off ({}); all speech goes to the cloud as before.'.format(exc))
        return None
    print('Local wake word: ON (Vosk). Outside a conversation nothing is sent until "Айсана".')
    return spotter
