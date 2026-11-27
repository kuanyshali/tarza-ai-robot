"""Spending meter: tokens -> dollars, per day file."""
import os
import tempfile
import unittest

import aisana_usage as usage


class UsageTests(unittest.TestCase):
    def setUp(self):
        self.original = usage.USAGE_DIR
        usage.USAGE_DIR = tempfile.mkdtemp()

    def tearDown(self):
        usage.USAGE_DIR = self.original

    def test_reply_and_hearing_cost(self):
        meter = usage.UsageMeter(save_every=0)
        meter.add_response({'input_tokens': 21000, 'output_tokens': 600, 'input_token_details': {
            'text_tokens': 5000, 'audio_tokens': 16000, 'cached_tokens': 14000,
            'cached_tokens_details': {'text_tokens': 4000, 'audio_tokens': 10000}},
            'output_token_details': {'text_tokens': 100, 'audio_tokens': 500}})
        meter.add_transcription(60)                       # one minute of talk heard
        meter.add_vision(3000, 100)
        s = usage.summary()
        p = usage.PRICES                                  # prices of the configured model
        expected = (1000 * p['text_in'] + 4000 * p['text_in_cached'] + 6000 * p['audio_in']
                    + 10000 * p['audio_in_cached'] + 100 * p['text_out'] + 500 * p['audio_out']) / 1e6
        self.assertAlmostEqual(s['replies_usd'], round(expected, 4), places=3)
        self.assertAlmostEqual(s['hearing_usd'], 0.0045, places=3)
        self.assertEqual(s['hearing_minutes'], 1.0)
        self.assertEqual((s['replies'], s['looks']), (1, 1))
        self.assertTrue(os.path.isfile(usage.path_for(usage.today())))   # the panel can read it

    def test_full_model_is_the_default(self):
        # Owner, Nov 27: the full realtime model, not mini; prices follow the model.
        self.assertEqual(usage.DEFAULT_REALTIME_MODEL, 'gpt-realtime-2.1')
        self.assertEqual(usage.MODEL_PRICES['gpt-realtime-2.1']['audio_out'], 64.00)
        self.assertEqual(usage.MODEL_PRICES['gpt-realtime-2.1-mini']['audio_out'], 20.00)

    def test_missing_usage_is_ignored(self):
        meter = usage.UsageMeter(save_every=0)
        meter.add_response(None)
        self.assertEqual(usage.summary()['usd'], 0)


if __name__ == '__main__':
    unittest.main()
