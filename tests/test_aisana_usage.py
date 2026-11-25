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
        # 1000 text*0.6 + 4000*0.06 + 6000 audio*10 + 10000*0.3 + 100*2.4 + 500*20 = 74080 per 1M
        self.assertAlmostEqual(s['replies_usd'], 0.074, places=3)
        self.assertAlmostEqual(s['hearing_usd'], 0.0045, places=3)
        self.assertEqual(s['hearing_minutes'], 1.0)
        self.assertEqual((s['replies'], s['looks']), (1, 1))
        self.assertTrue(os.path.isfile(usage.path_for(usage.today())))   # the panel can read it

    def test_missing_usage_is_ignored(self):
        meter = usage.UsageMeter(save_every=0)
        meter.add_response(None)
        self.assertEqual(usage.summary()['usd'], 0)


if __name__ == '__main__':
    unittest.main()
