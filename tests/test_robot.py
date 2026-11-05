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
    # ast.Module(type_ignores=...) was introduced after Jetson's Python 3.6.
    try:
        module = ast.Module(body=[node], type_ignores=[])
    except TypeError:
        module = ast.Module(body=[node])
    module = ast.fix_missing_locations(module)
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

    def test_xiaomi_device_follows_alsa_card_changes_and_ignores_kt(self):
        listing = (
            'card 0: tegrahda [tegra-hda], device 3: HDMI 0 [HDMI 0]\n'
            'card 2: Audio [KT USB Audio], device 0: USB Audio [USB Audio]\n'
            'card 5: Device [USB Composite Device], device 0: USB Audio [USB Audio]\n'
        )
        def run(*args, **kwargs):
            return SimpleNamespace(returncode=0, stdout=listing, stderr='')
        finder = isolated(
            'xiaomi_speaker_device', re=re,
            XIAOMI_ALSA_LABEL='USB Composite Device',
            subprocess=SimpleNamespace(run=run, PIPE=-1, TimeoutExpired=TimeoutError),
            AudioPlaybackError=RuntimeError
        )
        self.assertEqual(finder(), 'plughw:5,0')

    def test_xiaomi_absent_never_falls_back_to_kt(self):
        def run(*args, **kwargs):
            return SimpleNamespace(returncode=0,
                stdout='card 2: Audio [KT USB Audio], device 0: USB Audio [USB Audio]\n',
                stderr='')
        finder = isolated(
            'xiaomi_speaker_device', re=re,
            XIAOMI_ALSA_LABEL='USB Composite Device',
            subprocess=SimpleNamespace(run=run, PIPE=-1, TimeoutExpired=TimeoutError),
            AudioPlaybackError=RuntimeError
        )
        with self.assertRaisesRegex(RuntimeError, 'not detected'):
            finder()

    def test_xiaomi_multiple_matching_devices_refused(self):
        def run(*args, **kwargs):
            return SimpleNamespace(returncode=0,
                stdout=('card 3: Device [USB Composite Device], device 0: USB Audio [USB Audio]\n'
                        'card 4: Device [USB Composite Device], device 0: USB Audio [USB Audio]\n'),
                stderr='')
        finder = isolated(
            'xiaomi_speaker_device', re=re,
            XIAOMI_ALSA_LABEL='USB Composite Device',
            subprocess=SimpleNamespace(run=run, PIPE=-1, TimeoutExpired=TimeoutError),
            AudioPlaybackError=RuntimeError
        )
        with self.assertRaisesRegex(RuntimeError, 'Multiple'):
            finder()

    def test_audio_playback_only_uses_xiaomi(self):
        calls = []
        def run(args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(returncode=0, stderr=b'')
        playback = isolated(
            'play_pcm',
            xiaomi_speaker_device=lambda: 'plughw:7,0',
            subprocess=SimpleNamespace(run=run, PIPE=-1, DEVNULL=-3, TimeoutExpired=TimeoutError),
            AudioPlaybackError=RuntimeError,
        )
        self.assertEqual(playback(b'PCM', 24000, quiet=True), 'plughw:7,0')
        self.assertEqual(calls[0][1]['input'], b'PCM')
        self.assertEqual(calls[0][0][3], 'plughw:7,0')
        self.assertEqual(len(calls), 1)

    def test_streamed_tts_starts_before_download_finishes(self):
        events = []
        class AudioResponse:
            def iter_content(self, chunk_size):
                events.append('chunk1')
                yield b'A' * 50000
                events.append('chunk2')
                yield b'B' * 12000
            def close(self):
                events.append('closed')
        class FakeStdin:
            closed = False
            def write(self, data):
                events.append(('write', len(data)))
            def close(self):
                self.closed = True
        class FakeStderr:
            def read(self):
                return b''
            def close(self):
                pass
        class FakePlayer:
            def __init__(self, *args, **kwargs):
                events.append(('spawn', args[0][3]))
                self.stdin = FakeStdin()
                self.stderr = FakeStderr()
            def wait(self, timeout=None):
                return 0
            def poll(self):
                return 0
        stream = isolated(
            'stream_tts_to_xiaomi',
            subprocess=SimpleNamespace(Popen=FakePlayer, PIPE=-1, DEVNULL=-3,
                                        TimeoutExpired=TimeoutError),
            time=time, TTS_PREBUFFER_SECONDS=1.0, DEBUG_TIMING=False,
            AudioPlaybackError=RuntimeError, threading=__import__('threading'),
            BARGE_IN_ENABLED=False, requests=__import__('requests')
        )
        result = stream(AudioResponse(), 'plughw:5,0', time.monotonic())
        self.assertEqual(result, 'plughw:5,0')
        self.assertTrue(events.index(('spawn', 'plughw:5,0')) < events.index('chunk2'))
        self.assertIn(('write', 50000), events)
        self.assertIn(('write', 12000), events)
        self.assertEqual(events[-1], 'closed')

    def test_tts_streams_only_to_verified_xiaomi(self):
        calls = []
        class AudioResponse:
            def raise_for_status(self):
                pass
            def close(self):
                pass
        def post(url, **kwargs):
            self.assertIn('/audio/speech', url)
            self.assertTrue(kwargs['stream'])
            return AudioResponse()
        def stream(response, device, started, spoken_text=''):
            calls.append(device)
            return device
        speaker = isolated(
            'speak', requests=SimpleNamespace(post=post), HEADERS={},
            ROBOT_NAME='Садам', time=time,
            xiaomi_speaker_device=lambda: 'plughw:5,0',
            stream_tts_to_xiaomi=stream
        )
        self.assertEqual(speaker('Hello!'), 'plughw:5,0')
        self.assertEqual(calls, ['plughw:5,0'])

    def test_tts_no_xiaomi_does_not_request_audio(self):
        def no_request(*args, **kwargs):
            self.fail('TTS should not be called if Xiaomi is missing')
        def no_xiaomi():
            raise RuntimeError('Xiaomi missing')
        speaker = isolated(
            'speak',
            requests=SimpleNamespace(post=no_request),
            HEADERS={}, ROBOT_NAME='Садам',
            time=time, xiaomi_speaker_device=no_xiaomi,
            AudioPlaybackError=RuntimeError,
        )
        with self.assertRaisesRegex(RuntimeError, 'Xiaomi missing'):
            speaker('Hello!')

    def test_local_camera_route_multilingual_and_ambiguous(self):
        local = isolated('local_camera_route', re=re)
        for question in ('Как дела?', 'how are you?', 'thanks!', 'Сәлем!'):
            with self.subTest(question=question):
                self.assertIs(local(question), False)
        for question in ('Что у меня в руке?', 'what am I holding?',
                         'не көріп тұрсың?', 'شايف إيه قدامك؟',
                         'are you good? what am I holding?'):
            with self.subTest(question=question):
                self.assertIs(local(question), True)
        self.assertIsNone(local('какая погода сегодня?'))
        self.assertIs(local('а теперь?', previous_camera_question='Что видишь?'), True)

    def test_unfinished_sentence_rules(self):
        unfinished = isolated('utterance_may_continue', re=re)
        for sample in (
            'Sadam, tell me about this and',
            'Я хотел спросить, потому что...',
            'Что ты думаешь, если',
            'I want to know because.',
            'Мен ойлаймын, бірақ',
            'قول لي عشان',
        ):
            with self.subTest(sample=sample):
                self.assertTrue(unfinished(sample))
        for sample in ('Как дела?', 'What am I holding?', 'Сколько будет два плюс два?'):
            with self.subTest(sample=sample):
                self.assertFalse(unfinished(sample))

    def test_listen_complete_joins_incomplete_fragments(self):
        responses = iter(['Sadam, I wanted to ask because', 'I have a question.', ''])
        wait_calls = []
        def listen(wait_seconds=None):
            wait_calls.append(wait_seconds)
            return next(responses)
        joiner = isolated(
            'listen_complete', listen=listen,
            utterance_may_continue=isolated('utterance_may_continue', re=re),
            MAX_CONTINUATION_SEGMENTS=2, CONTINUATION_WAIT=2.5,
            SEMANTIC_END_ENABLED=False
        )
        self.assertEqual(joiner(wait_seconds=15), 
                         'Sadam, I wanted to ask because I have a question.')
        self.assertEqual(wait_calls, [15, 2.5])

    def test_followup_intent_rejects_ambient_and_keeps_short_answer(self):
        calls = []
        def post(url, **kwargs):
            calls.append(kwargs)
            return FakeResponse('NO')
        gate = isolated(
            'followup_is_addressed',
            re=re, requests=SimpleNamespace(post=post), HEADERS={},
            ROUTER_MODEL='gpt-6-luna',
            build_chat_request=isolated('build_chat_request', REASONING_EFFORT='none')
        )
        self.assertTrue(gate('Yes', 'Would you like me to continue?'))
        self.assertFalse(gate('Друг, закажи мне такси', 'I see a chair.'))
        self.assertEqual(len(calls), 1)

    def test_barge_in_accepts_only_explicit_wake_name(self):
        import threading
        class Player:
            stopped = False
            def poll(self):
                return -15 if self.stopped else None
            def terminate(self):
                self.stopped = True
        player = Player()
        stop = threading.Event()
        cancelled = threading.Event()
        worker = isolated(
            'barge_in_worker',
            record_until_silence=lambda **kwargs: True,
            transcribe_audio=lambda **kwargs: 'Садам, стоп',
            wake_question=self.wake,
            interrupt_lock=threading.Lock(),
            pending_interrupt=None
        )
        worker(player, stop, cancelled, 'Today is sunny.')
        self.assertTrue(cancelled.is_set())
        self.assertTrue(player.stopped)
        self.assertEqual(worker.__globals__['pending_interrupt'], 'Садам, стоп')

    def test_barge_in_ignores_unaddressed_background_voice(self):
        import threading
        class Player:
            times = 0
            def poll(self):
                self.times += 1
                return None if self.times < 3 else 0
            def terminate(self):
                raise AssertionError('Background speech must not stop TTS')
        player = Player()
        stop = threading.Event()
        cancelled = threading.Event()
        worker = isolated(
            'barge_in_worker',
            record_until_silence=lambda **kwargs: True,
            transcribe_audio=lambda **kwargs: 'Да, я понял тебя',
            wake_question=self.wake,
            interrupt_lock=threading.Lock(),
            pending_interrupt=None
        )
        worker(player, stop, cancelled, 'Some text')
        self.assertFalse(cancelled.is_set())
        self.assertIsNone(worker.__globals__['pending_interrupt'])

    def test_direct_arabic_question_does_not_need_intent_api(self):
        local = isolated('local_followup_intent', re=re)
        self.assertTrue(local('تقدر توصفلي اللي حصل في غزة ده إبادة جماعية ولا لأ؟'))
        self.assertTrue(local('А это правда?'))
        self.assertTrue(local('What happened yesterday?'))
        self.assertIsNone(local('Друг, я вернусь через минуту'))
        called = []
        def no_network(*args, **kwargs):
            called.append(True)
            raise AssertionError('Direct question must not call API')
        gate = isolated(
            'followup_is_addressed',
            re=re, local_followup_intent=local,
            requests=SimpleNamespace(post=no_network)
        )
        self.assertTrue(gate('تقدر توصفلي اللي حصل في غزة ده إبادة جماعية ولا لأ؟'))
        self.assertEqual(called, [])

    def test_ambiguous_intent_timeout_does_not_hang(self):
        local = isolated('local_followup_intent', re=re)
        calls = []
        def timeout_post(url, **kwargs):
            calls.append(kwargs)
            raise TimeoutError('request timed out')
        gate = isolated(
            'followup_is_addressed', re=re,
            local_followup_intent=local,
            requests=SimpleNamespace(post=timeout_post),
            HEADERS={}, ROUTER_MODEL='gpt-6-luna',
            build_chat_request=isolated('build_chat_request', REASONING_EFFORT='none')
        )
        self.assertFalse(gate('Друг, я вернусь через минуту', 'Hello'))
        self.assertEqual(calls[0]['timeout'], (3, 4))

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
        camera = isolated('needs_camera', local_camera_route=isolated('local_camera_route', re=re), requests=SimpleNamespace(post=post), HEADERS={}, time=time, DEBUG_TIMING=False, ROUTER_MODEL='gpt-6-luna', build_chat_request=isolated('build_chat_request', REASONING_EFFORT='none'))
        self.assertTrue(camera('что у меня в руке?'))
        self.assertFalse(camera('какая погода?'))
        self.assertTrue(camera('а в другой руке?', previous_camera_question='что у меня в руке?'))


if __name__ == '__main__':
    unittest.main()
