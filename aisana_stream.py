"""Optional streaming transcription from Jetson's ALSA microphone.

Uses OpenAI Realtime transcription sessions over WebSocket, with server-side VAD.
The microphone sends 24-kHz mono PCM continuously in 30-ms chunks; no WAV upload.
Python 3.6 compatible with websocket-client==1.3.1.
"""
import base64
import json
import os
import subprocess
import threading
import time


def stream_transcribe(mic_device, api_key, wait_seconds=None, stop_event=None):
    try:
        import websocket
    except ImportError:
        raise RuntimeError('Streaming mode needs websocket-client: .venv/bin/pip install websocket-client==1.3.1')
    url = 'wss://api.openai.com/v1/realtime?intent=transcription'
    ws = websocket.create_connection(
        url, timeout=10,
        header=['Authorization: Bearer ' + api_key]
    )
    rate = 24000
    chunk_size = 1440
    silence_ms = int(float(os.environ.get('AISANA_STREAM_SILENCE_MS', '2000')))
    silence_ms = max(500, min(5000, silence_ms))
    ws.send(json.dumps({
        'type': 'session.update',
        'session': {
            'type': 'transcription',
            'audio': {
                'input': {
                    'format': {'type': 'audio/pcm', 'rate': 24000},
                    'transcription': {
                        'model': os.environ.get('AISANA_STREAM_STT_MODEL', 'gpt-4o-mini-transcribe'),
                        'prompt': 'Aisana, Aysana, Айсана, Эйсана. Multilingual Russian, Kazakh, English and Arabic.'
                    },
                    'turn_detection': {
                        'type': 'server_vad',
                        'threshold': 0.5,
                        'prefix_padding_ms': 500,
                        'silence_duration_ms': silence_ms
                    }
                }
            }
        }
    }))

    done = threading.Event()
    errors = []
    started = threading.Event()
    process = None

    def send_microphone():
        try:
            while not done.is_set():
                if stop_event is not None and stop_event.is_set():
                    break
                pcm = process.stdout.read(chunk_size)
                if len(pcm) != chunk_size:
                    raise RuntimeError('Microphone stream ended unexpectedly')
                ws.send(json.dumps({
                    'type': 'input_audio_buffer.append',
                    'audio': base64.b64encode(pcm).decode('ascii')
                }))
        except Exception as exc:
            if not done.is_set():
                errors.append(str(exc))
                done.set()

    try:
        process = subprocess.Popen(
            ['arecord', '-q', '-D', mic_device, '-t', 'raw',
             '-f', 'S16_LE', '-r', str(rate), '-c', '1'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        worker = threading.Thread(target=send_microphone, daemon=True)
        worker.start()
        ws.settimeout(0.5)
        began = time.monotonic()
        print('Streaming microphone (KT):', mic_device)
        while not done.is_set():
            if stop_event is not None and stop_event.is_set():
                return ''
            if wait_seconds is not None and not started.is_set() and time.monotonic() - began > wait_seconds:
                return ''
            try:
                event = json.loads(ws.recv())
            except websocket.WebSocketTimeoutException:
                continue
            event_type = event.get('type', '')
            if event_type == 'input_audio_buffer.speech_started':
                started.set()
                print('Speech detected (streaming)...')
            elif event_type == 'input_audio_buffer.speech_stopped':
                print('Speech pause detected; awaiting transcription...')
            elif event_type == 'conversation.item.input_audio_transcription.completed':
                transcript = event.get('transcript', '').strip()
                if transcript:
                    print('Streaming transcription:', transcript)
                    return transcript
            elif event_type == 'error':
                raise RuntimeError('Realtime transcription: ' + str(event.get('error', {})))
        if errors:
            raise RuntimeError(errors[0])
        return ''
    finally:
        done.set()
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        try:
            ws.close()
        except Exception:
            pass
