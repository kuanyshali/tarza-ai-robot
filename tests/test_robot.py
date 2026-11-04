"""Offline smoke tests; no Jetson hardware, API key, or OpenAI calls required."""
import ast
import difflib
import io
import os
import pathlib
import re
import time
import tempfile
import unittest
import wave
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'robot.py').read_text(encoding='utf-8')
TREE = ast.parse(SOURCE)
FUNCTIONS = {node.name: node for node in TREE.body if isinstance(node, ast.FunctionDef)}


def isolated(name, **environment):
    node = FUNCTIONS[name]
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = dict(environment)
    exec(compile(module, str(ROOT / 'robot.py'), 'exec'), namespace)
    return namespace[name]


class FakeResponse:
    def __init__(self, value):
        self.value = value

    def raise_for_status(self):
        pass

    def json(self):
        return {'choices': [{'message': {'content': self.value}}]}


class RobotTests(unittest.TestCase):
    def setUp(self):
        words = ('садам', 'саддам', 'sadam', 'saddam', 'صدام', 'سدام', 'садом', 'саднам', 'سعدتم')
        self.wake = isolated('wake_question', re=re, WAKE_CANDIDATES=words)

    def test_wake_name_variants(self):
        for phrase in ('Садам, что ты видишь?', 'Садом, что ты видишь?',
                       'Саднам, что у меня в руке?', 'صدام، شايف إيه؟',
                       'سعدتم، تقدر تقولي؟', 'Sadam, can you see me?'):
            with self.subTest(phrase=phrase):
                self.assertIsNotNone(self.wake(phrase))

    def test_no_name(self):
        for phrase in ('что ты видишь?', 'تقدر تقولي مين فاز؟',
                       'Can you see the wall?', 'Какая погода?'):
            with self.subTest(phrase=phrase):
                self.assertIsNone(self.wake(phrase))

    def test_languages(self):
        hints = re.compile(r'\b(?:enta|inta|enti|ezayak|izzayak|shayef|shaif|shif|eih|delwaqt|dilwaqt|ana|mesh|mish|fein|fen|bet|bt)\b', re.I)
        language = isolated('response_language', re=re, ARABIC_LATIN_HINTS=hints)
        self.assertEqual(language('شايف إيه قدامك؟'), 'Egyptian Arabic')
        self.assertEqual(language('inta shayef eih'), 'Egyptian Arabic')
        self.assertEqual(language('что ты видишь'), 'Russian')
        self.assertEqual(language('не көріп тұрсың'), 'Kazakh')
        self.assertEqual(language('what do you see'), 'English')

    def test_wake_audio_prefix_keeps_first_2_7_seconds(self):
        prefix = isolated('wake_audio_prefix', wave=wave, io=io, os=os)
        with tempfile.TemporaryDirectory() as directory:
            filename = str(pathlib.Path(directory) / 'sample.wav')
            with wave.open(filename, 'wb') as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                writer.writeframes(b'\\x01\\x02' * (16000 * 5))
            clip = prefix(filename, seconds=2.7)
        with wave.open(io.BytesIO(clip), 'rb') as reader:
            self.assertEqual(reader.getframerate(), 16000)
            self.assertEqual(reader.getnframes(), int(16000 * 2.7))

    def test_wake_recovery_does_not_invent_name(self):
        normal_question = 'تقدر تقول لي مين فاز بالانتخابات؟'
        recovered = isolated(
            'recover_wake_question',
            transcribe_audio=lambda **kwargs: 'صدام، تقدر تقول لي؟',
            wake_question=self.wake, os=os
        )
        self.assertEqual(recovered(normal_question), normal_question)

        not_detected = isolated(
            'recover_wake_question',
            transcribe_audio=lambda **kwargs: 'تقدر تقول لي؟',
            wake_question=self.wake, os=os
        )
        self.assertIsNone(not_detected(normal_question))

    def test_transcription_sends_multilingual_wake_hints(self):
        captured = []
        class TranscriptResponse:
            def raise_for_status(self):
                pass
            def json(self):
                return {'text': 'صدام، تقدر تقول لي؟'}
        def post(url, **kwargs):
            captured.append(kwargs)
            return TranscriptResponse()
        transcribe = isolated(
            'transcribe_audio', requests=SimpleNamespace(post=post),
            HEADERS={}, TRANSCRIBE_MODEL='gpt-transcribe', DEBUG_TIMING=False,
            time=time, wake_audio_prefix=lambda filename, seconds: b'WAVE'
        )
        with tempfile.TemporaryDirectory() as directory:
            filename = str(pathlib.Path(directory) / 'example.wav')
            pathlib.Path(filename).write_bytes(b'WAVE')
            self.assertEqual(transcribe(filename), 'صدام، تقدر تقول لي؟')
            self.assertEqual(transcribe(filename, first_seconds=2.7),
                             'صدام، تقدر تقول لي؟')
        self.assertEqual(len(captured), 2)
        self.assertIn(('keywords[]', 'صدام'), captured[0]['data'])
        self.assertIn(('languages[]', 'ar'), captured[0]['data'])
        self.assertEqual(captured[1]['files']['file'][0], 'wake_prefix.wav')
        self.assertEqual(captured[1]['files']['file'][1], b'WAVE')

    def test_alsa_scan_prefers_usb_speaker_even_if_card_number_changes(self):
        listing = (
            'card 0: tegrahda [tegra-hda], device 3: HDMI 0 [HDMI 0]\n'
            'card 2: Adapter [USB Audio Adapter], device 0: USB Audio [USB Audio]\n'
            'card 4: Speaker [USB Speaker], device 0: USB Audio [USB Audio]\n'
        )
        def run(*args, **kwargs):
            return SimpleNamespace(returncode=0, stdout=listing, stderr='')
        candidates = isolated(
            'audio_output_candidates',
            os=SimpleNamespace(environ={}), re=re,
            subprocess=SimpleNamespace(run=run, PIPE=-1, TimeoutExpired=TimeoutError),
        )
        self.assertEqual(candidates(), ['plughw:4,0', 'plughw:2,0'])

    def test_alsa_device_override(self):
        candidates = isolated(
            'audio_output_candidates',
            os=SimpleNamespace(environ={'SADAM_SPEAKER_DEVICE': 'plughw:CARD=Speaker,DEV=0'}),
        )
        self.assertEqual(candidates(), ['plughw:CARD=Speaker,DEV=0'])

    def test_audio_playback_falls_back_on_alsa_error(self):
        calls = []
        def run(args, **kwargs):
            calls.append((args, kwargs))
            if len(calls) == 1:
                return SimpleNamespace(returncode=1, stderr=b'Invalid value for card')
            return SimpleNamespace(returncode=0, stderr=b'')
        playback = isolated(
            'play_pcm',
            audio_output_candidates=lambda: ['plughw:3,0', 'plughw:4,0'],
            subprocess=SimpleNamespace(run=run, PIPE=-1, DEVNULL=-3, TimeoutExpired=TimeoutError),
            AudioPlaybackError=RuntimeError,
        )
        self.assertEqual(playback(b'PCM', 24000, quiet=True), 'plughw:4,0')
        self.assertEqual(calls[0][1]['input'], b'PCM')
        self.assertEqual(calls[1][0][3], 'plughw:4,0')

    def test_tts_buffers_pcm_before_playback(self):
        pcm = b'\\x01\\x02' * 24000
        calls = []
        class AudioResponse:
            content = pcm
            def raise_for_status(self):
                pass

        def post(url, **kwargs):
            self.assertIn('/audio/speech', url)
            self.assertNotIn('stream', kwargs)
            return AudioResponse()

        def play(data, rate):
            calls.append((data, rate))
            return 'plughw:4,0'

        speaker = isolated(
            'speak',
            requests=SimpleNamespace(post=post),
            HEADERS={},
            ROBOT_NAME='Садам',
            DEBUG_TIMING=False,
            time=time,
            audio_output_candidates=lambda: ['plughw:4,0'],
            play_pcm=play,
            AudioPlaybackError=RuntimeError,
        )
        speaker('Hello!')
        self.assertEqual(calls, [(pcm, 24000)])

    def test_tts_no_speaker_does_not_request_audio(self):
        def no_request(*args, **kwargs):
            self.fail('TTS should not be called if there is no output')
        speaker = isolated(
            'speak',
            requests=SimpleNamespace(post=no_request),
            HEADERS={}, ROBOT_NAME='Садам', DEBUG_TIMING=False,
            time=time, audio_output_candidates=lambda: [],
            AudioPlaybackError=RuntimeError,
        )
        with self.assertRaises(RuntimeError):
            speaker('Hello!')

    def test_new_model_payload(self):
        builder = isolated('build_chat_request', REASONING_EFFORT='none')
        payload = builder('gpt-6.1-terra', [{'role': 'user', 'content': 'hello'}], 500)
        self.assertEqual(payload['model'], 'gpt-6.1-terra')
        self.assertEqual(payload['reasoning_effort'], 'none')
        self.assertNotIn('temperature', payload)
        self.assertEqual(payload['max_completion_tokens'], 500)
        legacy = builder('gpt-4o-mini', [], 32, 0)
        self.assertEqual(legacy['temperature'], 0)

    def test_camera_router(self):
        seen = []
        def post(url, **kwargs):
            seen.append(kwargs['json']['messages'][-1]['content'])
            return FakeResponse('YES' if 'что у меня в руке?' in seen[-1] else 'NO')
        camera = isolated('needs_camera', requests=SimpleNamespace(post=post), HEADERS={}, time=time, DEBUG_TIMING=False, ROUTER_MODEL='gpt-6-luna', build_chat_request=isolated('build_chat_request', REASONING_EFFORT='none'))
        self.assertTrue(camera('что у меня в руке?'))
        self.assertFalse(camera('какая погода?'))
        self.assertTrue(camera('а в другой руке?', previous_camera_question='что у меня в руке?'))


if __name__ == '__main__':
    unittest.main()
