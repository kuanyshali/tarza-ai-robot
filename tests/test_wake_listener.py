"""Offline Vosk wake-listener tests; no microphone or model download needed."""
import json
import unittest
import wake_listener as wake


class WakeListenerTests(unittest.TestCase):
    def test_local_wake_variants(self):
        for phrase in ('садам', 'саддам', 'садам какая погода',
                       'sadam what time', 'saddam hello', 'صدام'):
            with self.subTest(phrase=phrase):
                self.assertTrue(wake.has_wake_word(phrase))

    def test_unrelated_speech_ignored(self):
        for phrase in ('', 'what a beautiful day', 'как дела у тебя',
                       'поговорим про садам позже', 'hey sadam', 'салам всем'):
            with self.subTest(phrase=phrase):
                self.assertFalse(wake.has_wake_word(phrase))

    def test_vosk_json_formats(self):
        self.assertEqual(
            wake.extract_recognition_text('{"partial":"садам привет"}'),
            'садам привет')
        self.assertEqual(
            wake.extract_recognition_text('{"text":"саддам"}'),
            'саддам')
        self.assertEqual(wake.extract_recognition_text('{not json'), '')

    def test_model_missing_error_is_actionable(self):
        with self.assertRaisesRegex(RuntimeError, 'SADAM_WAKE_MODE=legacy'):
            wake.load_vosk_model('/path/that/does/not/exist')


if __name__ == '__main__':
    unittest.main()
