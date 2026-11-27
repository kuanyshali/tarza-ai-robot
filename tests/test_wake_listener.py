"""Offline Vosk wake-listener tests; no microphone or model download needed."""
import json
import unittest
import wake_listener as wake


class WakeListenerTests(unittest.TestCase):
    def test_local_wake_variants(self):
        for phrase in ('айсана', 'эйсана', 'айсана какая погода',
                       'aisana what time', 'aysana hello', 'آيسانا'):
            with self.subTest(phrase=phrase):
                self.assertTrue(wake.has_wake_word(phrase))

    def test_unrelated_speech_ignored(self):
        for phrase in ('', 'what a beautiful day', 'как дела у тебя',
                       'поговорим про айсану позже', 'hey aisana', 'салам всем',
                       'садам'):
            with self.subTest(phrase=phrase):
                self.assertFalse(wake.has_wake_word(phrase))

    def test_vosk_json_formats(self):
        self.assertEqual(
            wake.extract_recognition_text('{"partial":"айсана привет"}'),
            'айсана привет')
        self.assertEqual(
            wake.extract_recognition_text('{"text":"эйсана"}'),
            'эйсана')
        self.assertEqual(wake.extract_recognition_text('{not json'), '')

    def test_model_missing_error_is_actionable(self):
        with self.assertRaisesRegex(RuntimeError, 'SADAM_WAKE_MODE=legacy'):
            wake.load_vosk_model('/path/that/does/not/exist')


if __name__ == '__main__':
    unittest.main()


class SpotterTests(unittest.TestCase):
    def word(self, text, conf, start, end):
        return {'word': text, 'conf': conf, 'start': start, 'end': end}

    def test_how_vosk_hears_the_name(self):
        # Measured with synthetic speech and vosk-model-small-ru-0.22.
        self.assertTrue(wake.name_in_words([self.word('ай', 1.0, 0.5, 0.7), self.word('сана', 1.0, 0.7, 1.1)]))
        self.assertFalse(wake.name_in_words([self.word('а', 1.0, 0.5, 0.6), self.word('сана', 1.0, 0.6, 1.0)]))  # "а сам"
        self.assertFalse(wake.name_in_words([self.word('ай', 0.88, 0.5, 0.7), self.word('сана', 1.0, 0.7, 1.0)]))  # "айфон"
        self.assertFalse(wake.name_in_words([self.word('ай', 1.0, 0.5, 0.7), self.word('сана', 1.0, 1.5, 1.9)]))   # apart
        self.assertFalse(wake.name_in_words([]))

    def test_spotter_resamples_and_reads_words_at_a_pause(self):
        class FakeRecognizer(object):
            def __init__(self):
                self.fed, self.final, self.words = [], False, []

            def AcceptWaveform(self, pcm):
                self.fed.append(len(pcm))
                return self.final

            def Result(self):
                return json.dumps({'result': self.words})
        recognizer = FakeRecognizer()
        spotter = wake.WakeSpotter(None, rate=24000, recognizer=recognizer)
        self.assertFalse(spotter.feed(b'\x00\x00' * 720))           # 30 ms at 24 kHz
        self.assertAlmostEqual(recognizer.fed[0], 960, delta=4)        # -> 30 ms at 16 kHz
        recognizer.final = True
        recognizer.words = [self.word('ай', 1.0, 0.1, 0.3), self.word('сана', 0.99, 0.3, 0.7)]
        self.assertTrue(spotter.feed(b'\x00\x00' * 720))

    def test_off_unless_asked(self):
        import os
        os.environ.pop('AISANA_LOCAL_WAKE', None)
        self.assertIsNone(wake.spotter_from_environment())
