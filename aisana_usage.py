"""What Aisana spends on OpenAI today: tokens, minutes and an estimate in dollars.

aisana_realtime adds every reply's usage (response.done), every transcription
and every camera look; the totals are kept per day in
~/.local/share/aisana/usage-YYYY-MM-DD.json, so the phone panel can show them
even when Aisana is off.

Prices are USD per 1M tokens (per minute for transcription), as listed in
November 2026; override with AISANA_PRICE_<NAME>=<number> if OpenAI changes them.
The estimate is for guidance only: the OpenAI Usage page is the bill.

Python 3.6, stdlib only.
"""
import json
import os
import threading
import time

USAGE_DIR = os.environ.get('AISANA_USAGE_DIR', os.path.expanduser('~/.local/share/aisana'))

# The realtime model (owner, Nov 27: the full model, not mini). Override with
# AISANA_REALTIME_MODEL; the prices below follow it.
DEFAULT_REALTIME_MODEL = 'gpt-realtime-2.1'
REALTIME_MODEL = os.environ.get('AISANA_REALTIME_MODEL', DEFAULT_REALTIME_MODEL)
MODEL_PRICES = {
    # per 1M tokens, OpenAI price list (Nov 2026)
    'gpt-realtime-2.1': {'text_in': 4.00, 'text_in_cached': 0.40, 'text_out': 24.00,
                         'audio_in': 32.00, 'audio_in_cached': 0.40, 'audio_out': 64.00,
                         'image_in': 5.00},
    'gpt-realtime-2.1-mini': {'text_in': 0.60, 'text_in_cached': 0.06, 'text_out': 2.40,
                              'audio_in': 10.00, 'audio_in_cached': 0.30, 'audio_out': 20.00,
                              'image_in': 0.80},
}
PRICES = dict(MODEL_PRICES.get(REALTIME_MODEL, MODEL_PRICES[DEFAULT_REALTIME_MODEL]))
PRICES.update({
    # transcription, per minute of audio
    'stt_minute': 0.0045,          # gpt-transcribe
    # camera answers (gpt-4o-mini), per 1M tokens
    'vision_in': 0.15, 'vision_out': 0.60,
})
for _name in list(PRICES):
    _value = os.environ.get('AISANA_PRICE_' + _name.upper())
    if _value:
        try:
            PRICES[_name] = float(_value)
        except ValueError:
            pass

FIELDS = ('text_in', 'text_in_cached', 'text_out', 'audio_in', 'audio_in_cached', 'audio_out',
          'image_in', 'stt_seconds', 'vision_in', 'vision_out', 'replies', 'transcripts', 'looks')


def today(now=None):
    return time.strftime('%Y-%m-%d', time.localtime(now))


def path_for(day):
    return os.path.join(USAGE_DIR, 'usage-{}.json'.format(day))


def load(day=None):
    totals = dict.fromkeys(FIELDS, 0)
    try:
        with open(path_for(day or today())) as handle:
            saved = json.load(handle)
        for key in FIELDS:
            totals[key] = saved.get(key, 0)
    except (IOError, OSError, ValueError):
        pass
    return totals


def cost(totals):
    """Dollars by part: replies (realtime model), hearing (transcription), camera."""
    m = 1e6
    replies = (totals['text_in'] * PRICES['text_in'] + totals['text_in_cached'] * PRICES['text_in_cached']
               + totals['text_out'] * PRICES['text_out'] + totals['audio_in'] * PRICES['audio_in']
               + totals['audio_in_cached'] * PRICES['audio_in_cached'] + totals['audio_out'] * PRICES['audio_out']
               + totals['image_in'] * PRICES['image_in']) / m
    hearing = totals['stt_seconds'] / 60.0 * PRICES['stt_minute']
    camera = (totals['vision_in'] * PRICES['vision_in'] + totals['vision_out'] * PRICES['vision_out']) / m
    return {'replies': replies, 'hearing': hearing, 'camera': camera, 'total': replies + hearing + camera}


def summary(day=None):
    totals = load(day)
    dollars = cost(totals)
    return {
        'usd': round(dollars['total'], 3),
        'replies_usd': round(dollars['replies'], 3),
        'hearing_usd': round(dollars['hearing'], 3),
        'camera_usd': round(dollars['camera'], 3),
        'hearing_minutes': round(totals['stt_seconds'] / 60.0, 1),
        'audio_in_k': round((totals['audio_in'] + totals['audio_in_cached']) / 1000.0, 1),
        'replies': totals['replies'],
        'looks': totals['looks'],
    }


class UsageMeter(object):
    """Thread-safe running totals for today, saved at most every few seconds."""

    def __init__(self, save_every=20.0):
        self.lock = threading.Lock()
        self.day = today()
        self.totals = load(self.day)
        self.save_every = save_every
        self.saved_at = time.monotonic()

    def _roll(self):
        if today() != self.day:               # past midnight: a new file
            self.save()
            self.day = today()
            self.totals = load(self.day)

    def add_response(self, usage):
        """usage from a Realtime response.done."""
        if not usage:
            return
        details = usage.get('input_token_details') or {}
        cached = details.get('cached_tokens_details') or {}
        out = usage.get('output_token_details') or {}
        cached_text = cached.get('text_tokens', 0) or 0
        cached_audio = cached.get('audio_tokens', 0) or 0
        if not cached and details.get('cached_tokens'):
            cached_text = details.get('cached_tokens') or 0     # older shape: no split
        with self.lock:
            self._roll()
            t = self.totals
            t['text_in'] += max(0, (details.get('text_tokens', 0) or 0) - cached_text)
            t['text_in_cached'] += cached_text
            t['audio_in'] += max(0, (details.get('audio_tokens', 0) or 0) - cached_audio)
            t['audio_in_cached'] += cached_audio
            t['image_in'] += details.get('image_tokens', 0) or 0
            t['text_out'] += out.get('text_tokens', 0) or 0
            t['audio_out'] += out.get('audio_tokens', 0) or 0
            t['replies'] += 1
        self._maybe_save()

    def add_transcription(self, seconds):
        with self.lock:
            self._roll()
            self.totals['stt_seconds'] += max(0.0, float(seconds or 0))
            self.totals['transcripts'] += 1
        self._maybe_save()

    def add_vision(self, prompt_tokens, completion_tokens):
        with self.lock:
            self._roll()
            self.totals['vision_in'] += prompt_tokens or 0
            self.totals['vision_out'] += completion_tokens or 0
            self.totals['looks'] += 1
        self._maybe_save()

    def _maybe_save(self):
        if time.monotonic() - self.saved_at >= self.save_every:
            self.save()

    def save(self):
        with self.lock:
            data = dict(self.totals, day=self.day)
            self.saved_at = time.monotonic()
        try:
            if not os.path.isdir(USAGE_DIR):
                os.makedirs(USAGE_DIR)
            tmp = path_for(data['day']) + '.part'
            with open(tmp, 'w') as handle:
                json.dump(data, handle)
            os.rename(tmp, path_for(data['day']))
        except (IOError, OSError) as exc:
            print('Usage not saved:', exc)
