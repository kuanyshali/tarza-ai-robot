"""Offline tests for automatic KT USB microphone discovery on Jetson.

Do not open the physical microphone, connect to a network, or require OpenAI.
Compatible with Jetson's Python 3.6.
"""
import ast
import pathlib
import re
import unittest
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / 'robot.py').read_text(encoding='utf-8'))


def capture_finder(listing, exit_code=0, error=''):
    node = next(node for node in TREE.body
                if isinstance(node, ast.FunctionDef)
                and node.name == 'kt_microphone_device')
    try:
        module = ast.Module(body=[node], type_ignores=[])
    except TypeError:
        module = ast.Module(body=[node])
    ast.fix_missing_locations(module)

    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=exit_code, stdout=listing, stderr=error)

    namespace = {
        'subprocess': SimpleNamespace(run=run, PIPE=-1, TimeoutExpired=TimeoutError),
        're': re,
        'KT_MIC_ALSA_LABEL': 'KT USB Audio',
    }
    exec(compile(module, str(ROOT / 'robot.py'), 'exec'), namespace)
    return namespace['kt_microphone_device'], calls


class KtMicrophoneTests(unittest.TestCase):
    def test_kt_at_card_three_after_usb_reconnect(self):
        finder, calls = capture_finder(
            'card 2: Device [USB Composite Device], device 0: USB Audio [USB Audio]\n'
            'card 3: Audio [KT USB Audio], device 0: USB Audio [USB Audio]\n'
        )
        self.assertEqual(finder(), 'plughw:3,0')
        self.assertEqual(calls, [['arecord', '-l']])

    def test_kt_at_card_two_before_usb_reconnect(self):
        finder, _ = capture_finder(
            'card 2: Audio [KT USB Audio], device 0: USB Audio [USB Audio]\n'
            'card 3: Device [USB Composite Device], device 0: USB Audio [USB Audio]\n'
        )
        self.assertEqual(finder(), 'plughw:2,0')

    def test_no_kt_fails_closed(self):
        finder, _ = capture_finder(
            'card 2: Device [USB Composite Device], device 0: USB Audio [USB Audio]\n'
        )
        with self.assertRaisesRegex(RuntimeError, 'Expected one KT USB Audio'):
            finder()

    def test_multiple_kt_endpoints_fails_closed(self):
        finder, _ = capture_finder(
            'card 2: Audio [KT USB Audio], device 0: USB Audio [USB Audio]\n'
            'card 3: Audio [KT USB Audio], device 0: USB Audio [USB Audio]\n'
        )
        with self.assertRaisesRegex(RuntimeError, 'found 2'):
            finder()

    def test_arecord_error_is_reported(self):
        finder, _ = capture_finder('', exit_code=1, error='ALSA unavailable')
        with self.assertRaisesRegex(RuntimeError, 'ALSA unavailable'):
            finder()

    def test_legacy_and_vosk_use_dynamic_microphone(self):
        definitions = {
            node.name: node for node in TREE.body
            if isinstance(node, ast.FunctionDef)
        }
        legacy = definitions['record_until_silence']
        main = definitions['main']
        self.assertTrue(any(
            isinstance(node, ast.Call) and
            isinstance(node.func, ast.Name) and
            node.func.id == 'kt_microphone_device'
            for node in ast.walk(legacy)
        ))
        wake_calls = [
            node for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == 'record_wake_phrase'
        ]
        self.assertEqual(len(wake_calls), 1)
        self.assertTrue(any(
            kw.arg == 'mic_device'
            and isinstance(kw.value, ast.Call)
            and isinstance(kw.value.func, ast.Name)
            and kw.value.func.id == 'kt_microphone_device'
            for kw in wake_calls[0].keywords
        ))


if __name__ == '__main__':
    unittest.main()
