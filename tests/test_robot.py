"""Offline smoke tests; no Jetson hardware, API key, or OpenAI calls required."""
import ast
import difflib
import pathlib
import re
import time
import unittest
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

        def play(args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(returncode=0)

        speaker = isolated(
            'speak',
            requests=SimpleNamespace(post=post),
            subprocess=SimpleNamespace(run=play, DEVNULL=-3),
            HEADERS={},
            ROBOT_NAME='Садам',
            DEBUG_TIMING=False,
            time=time
        )
        speaker('Hello!')
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]['input'], pcm)
        self.assertEqual(calls[0][0][0], 'aplay')

    def test_camera_router(self):
        seen = []
        def post(url, **kwargs):
            seen.append(kwargs['json']['messages'][-1]['content'])
            return FakeResponse('YES' if 'что у меня в руке?' in seen[-1] else 'NO')
        camera = isolated('needs_camera', requests=SimpleNamespace(post=post), HEADERS={}, time=time, DEBUG_TIMING=False)
        self.assertTrue(camera('что у меня в руке?'))
        self.assertFalse(camera('какая погода?'))
        self.assertTrue(camera('а в другой руке?', previous_camera_question='что у меня в руке?'))


if __name__ == '__main__':
    unittest.main()
