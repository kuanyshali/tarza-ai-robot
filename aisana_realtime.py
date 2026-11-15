"""Aisana experimental full-duplex speech-to-speech (Jetson Nano, Python 3.6).

Separate from robot.py: no camera/Arduino control, no always-on background service.
Uses websocket-client==1.3.1, arecord, aplay and Realtime GA.
Ctrl+C stops all subprocesses. Prefer headphones for first interruption tests.
"""
import base64
import json
import os
import re
import subprocess
import threading
import time

import websocket

RATE = 24000
CHUNK = 1440  # 30 ms of mono S16_LE PCM
MODEL = os.environ.get('AISANA_REALTIME_MODEL', 'gpt-realtime-2.1-mini')
VOICE = os.environ.get('AISANA_REALTIME_VOICE', 'marin')
BARGE_IN = os.environ.get('AISANA_REALTIME_BARGE_IN', '0') == '1'
SILENCE_MS = int(os.environ.get('AISANA_REALTIME_SILENCE_MS', '850'))
KEY = os.environ.get('OPENAI_API_KEY')
PROMPT = (
    'You are Aisana (Айсана), a warm, witty, helpful female-voiced robot. '
    'Speak naturally in the language of the user: Russian, Kazakh or English. '
    'In Russian refer to yourself in feminine forms (я поняла, я готова). '
    'Reply conversationally and concisely, usually one to three sentences. '
    'The user may say Айсана and pause before continuing; wait for the whole question. '
    'Do not claim to see anything; camera tools are not connected in this mode. '
    'Do not claim to control physical motors or perform actions. '
    'Do not invent live information. '
)


def alsa_device(command, expected, description):
    output = subprocess.check_output([command, '-l'], stderr=subprocess.STDOUT).decode(
        'utf-8', errors='replace')
    matches = []
    for line in output.splitlines():
        m = re.search(r'^card\s+(\d+):\s*[^\[]+\[([^\]]+)\],\s*device\s+(\d+):', line)
        if m and expected.lower() in m.group(2).lower():
            matches.append('plughw:{},{}'.format(m.group(1), m.group(3)))
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise RuntimeError('{} must match exactly one ALSA device; found {}.\n{}'.format(
            description, matches, output))
    return matches[0]


class RealtimeAisana(object):
    def __init__(self):
        self.ws = None
        self.send_lock = threading.Lock()
        self.play_lock = threading.Lock()
        self.stop = threading.Event()
        self.mic = None
        self.speaker = None
        self.playback = None
        self.responding = False
        self.playing = False
        self.speech_started_at = 0
        self.playback_interrupted = False

    def send(self, payload):
        with self.send_lock:
            self.ws.send(json.dumps(payload))

    def start_playback(self):
        self.stop_playback()
        with self.play_lock:
            self.playback = subprocess.Popen(
                ['aplay', '-q', '-D', self.speaker, '-t', 'raw', '-f', 'S16_LE',
                 '-r', str(RATE), '-c', '1'],
                stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self.playing = True

    def stop_playback(self):
        with self.play_lock:
            p = self.playback
            self.playback = None
            self.playing = False
            if p:
                try:
                    p.terminate()
                    p.wait(timeout=1)
                except Exception:
                    try:
                        p.kill()
                        p.wait()
                    except Exception:
                        pass

    def capture(self):
        try:
            self.mic = subprocess.Popen(
                ['arecord', '-q', '-D', self.mic_device, '-t', 'raw',
                 '-f', 'S16_LE', '-r', str(RATE), '-c', '1'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            while not self.stop.is_set():
                pcm = self.mic.stdout.read(CHUNK)
                if len(pcm) != CHUNK:
                    raise RuntimeError('Microphone capture ended unexpectedly')
                self.send({'type': 'input_audio_buffer.append',
                           'audio': base64.b64encode(pcm).decode('ascii')})
        except Exception as exc:
            if not self.stop.is_set():
                print('Microphone/connection error:', exc)
                self.stop.set()

    def handle(self, e):
        typ = e.get('type', '')
        if typ == 'session.updated':
            print('Realtime session ready. Talk to Aisana (Ctrl+C to stop).')
        elif typ == 'input_audio_buffer.speech_started':
            self.speech_started_at = time.monotonic()
            print('You: speaking...')
            if BARGE_IN and (self.responding or self.playing):
                self.playback_interrupted = True
                self.stop_playback()
                # Realtime server_vad with interrupt_response=True already
                # cancels the active generation. A second response.cancel races
                # with response.done and triggers response_cancel_not_active.
        elif typ == 'response.created':
            self.responding = True
            self.playback_interrupted = False
        elif typ == 'response.output_audio.delta':
            if self.playback_interrupted:
                return
            if self.playback is None:
                self.start_playback()
                print('Aisana: speaking...')
            chunk = base64.b64decode(e.get('delta', ''))
            with self.play_lock:
                if self.playback and self.playback.stdin:
                    try:
                        self.playback.stdin.write(chunk)
                        self.playback.stdin.flush()
                    except (IOError, OSError, ValueError):
                        pass
        elif typ == 'response.output_audio_transcript.delta':
            print(e.get('delta', ''), end='', flush=True)
        elif typ == 'response.output_audio_transcript.done':
            print('')
        elif typ == 'response.done':
            self.responding = False
            # Drain already-buffered audio rather than cutting off the voice.
            # Never block the websocket receive loop while waiting for playback.
            with self.play_lock:
                p = self.playback
                self.playback = None
                self.playing = False
            if p:
                try:
                    p.stdin.close()
                except Exception:
                    pass
                def finish_playback(proc):
                    try:
                        proc.wait(timeout=20)
                    except Exception:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                threading.Thread(target=finish_playback, args=(p,), daemon=True).start()
        elif typ == 'error':
            print('Realtime API error:', e.get('error', {}))

    def run(self):
        if not KEY:
            raise RuntimeError('OPENAI_API_KEY is missing')
        self.mic_device = alsa_device('arecord', 'KT USB Audio', 'KT microphone')
        self.speaker = alsa_device('aplay', 'USB Composite Device', 'Xiaomi speaker')
        print('Input:', self.mic_device, 'Output:', self.speaker)
        print('Model:', MODEL, 'Voice:', VOICE,
              'Barge-in:', 'ON (experimental)' if BARGE_IN else 'OFF')
        self.ws = websocket.create_connection(
            'wss://api.openai.com/v1/realtime?model=' + MODEL,
            timeout=10, header=['Authorization: Bearer ' + KEY])
        try:
            self.send({
                'type': 'session.update',
                'session': {
                    'type': 'realtime',
                    'model': MODEL,
                    'instructions': PROMPT,
                    'output_modalities': ['audio'],
                    'audio': {
                        'input': {
                            'format': {'type': 'audio/pcm', 'rate': RATE},
                            'turn_detection': {
                                'type': 'server_vad',
                                'threshold': 0.5,
                                'prefix_padding_ms': 500,
                                'silence_duration_ms': max(500, min(3000, SILENCE_MS)),
                                'create_response': True,
                                'interrupt_response': BARGE_IN
                            }
                        },
                        'output': {
                            'format': {'type': 'audio/pcm', 'rate': RATE},
                            'voice': VOICE
                        }
                    }
                }
            })
            self.ws.settimeout(0.5)
            worker = threading.Thread(target=self.capture, daemon=True)
            worker.start()
            while not self.stop.is_set():
                try:
                    event = json.loads(self.ws.recv())
                except websocket.WebSocketTimeoutException:
                    continue
                self.handle(event)
        finally:
            self.stop.set()
            if self.mic:
                self.mic.terminate()
                try:
                    self.mic.wait(timeout=2)
                except Exception:
                    self.mic.kill()
                    self.mic.wait()
            self.stop_playback()
            if self.ws:
                self.ws.close()


if __name__ == '__main__':
    try:
        RealtimeAisana().run()
    except KeyboardInterrupt:
        print('\nAisana stopped.')
    except Exception as exc:
        print('Aisana realtime error:', exc)
        raise
