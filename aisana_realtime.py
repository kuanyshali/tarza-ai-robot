"""Aisana realtime speech-to-speech (Jetson Nano, Python 3.6).

Fast path: microphone PCM streams to the OpenAI Realtime API and spoken audio
streams back, without separate STT / chat / TTS round trips.
Tools: fresh camera frame, Arm team's Arduino arms over USB, live weather and
web search. A conversation starts with the name "Aisana" (AISANA_REALTIME_WAKE=1,
default); follow-ups need no name for AISANA_DIALOG_TIMEOUT seconds.
Uses websocket-client==1.3.1, arecord, aplay and Realtime GA.
Ctrl+C stops all subprocesses. Prefer headphones for first interruption tests.
"""
import audioop
import base64
import collections
import json
import os
import queue
import requests
import re
import signal
import subprocess
import sys
import threading
import time

try:
    import websocket
except ImportError:          # tests and robot.py do not need websocket-client
    websocket = None

from aisana_arm import (connect_from_environment, parse_arm_commands, make_intent, describe,
                        ARM_WORDS, UP_WORDS, DOWN_WORDS, CONTINUE_WORDS, NUMBER_ANY,
                        MOVE_WORDS, OPEN_WORDS, CLOSE_WORDS)
from sadam_features import WEB_ENABLED, get_weather, search_web
import aisana_legs
from aisana_legs import parse_legs_command, LEGS_ACTIONS, legs_reply
from aisana_arm import arm_reply
import aisana_faces
import aisana_memory
import wake_listener
from aisana_faces import parse_memory_command

RATE = 24000
CHUNK = 1440  # 30 ms of mono S16_LE PCM
MODEL = os.environ.get('AISANA_REALTIME_MODEL', 'gpt-realtime-2.1-mini')
VOICE = os.environ.get('AISANA_REALTIME_VOICE', 'marin')
BARGE_IN = os.environ.get('AISANA_REALTIME_BARGE_IN', '0') == '1'
SILENCE_MS = int(os.environ.get('AISANA_REALTIME_SILENCE_MS', '700'))
ECHO_GUARD = os.environ.get('AISANA_ECHO_GUARD', '1') == '1'
ECHO_TAIL_SECONDS = float(os.environ.get('AISANA_ECHO_TAIL_SECONDS', '1.0'))
# The Xiaomi USB speaker drops the first ~0.2 s after the stream opens: lead with silence.
PLAYBACK_LEAD_MS = max(0, min(1000, int(os.environ.get('AISANA_PLAYBACK_LEAD_MS', '300'))))
KEY = os.environ.get('OPENAI_API_KEY')
# gpt-transcribe: OpenAI's newer, more accurate model for committed-turn realtime
# transcription (uses earlier turns as context). Falls back by itself if refused.
STT_MODEL = os.environ.get('AISANA_REALTIME_STT_MODEL', 'gpt-transcribe')
STT_FALLBACK = 'gpt-4o-mini-transcribe'
# ALSA label of the microphone (arecord -l); still exactly one device, no fallback.
# For the ReSpeaker USB Mic Array: AISANA_MIC_LABEL=ReSpeaker
MIC_LABEL = os.environ.get('AISANA_MIC_LABEL', 'KT USB Audio')
# She speaks only this language for now (Nov 23: English for the presentation).
# AISANA_LANGUAGE=auto = answer in the language of the user's words, as before.
REPLY_LANGUAGE = os.environ.get('AISANA_LANGUAGE', 'English')
FIXED_LANGUAGE = None if REPLY_LANGUAGE.lower() == 'auto' else REPLY_LANGUAGE
# After a guest with a title is seen: polite only, no sarcasm, for this long.
# Off by default (Nov 23: "верни сарказм и шутки как раньше"); e.g. 15 to turn it on.
VIP_MINUTES = float(os.environ.get('AISANA_VIP_MINUTES', '0'))
MIC_GAIN = float(os.environ.get('AISANA_MIC_GAIN', '1.0'))          # software gain, e.g. 2.0
MIC_LEVEL = os.environ.get('AISANA_MIC_LEVEL', '0') == '1'           # print loudness once a second
VAD_THRESHOLD = float(os.environ.get('AISANA_VAD_THRESHOLD', '0.5'))  # lower = hears quieter speech
# far_field: the speaker stands a metre or more from the robot's microphone.
NOISE_REDUCTION = os.environ.get('AISANA_NOISE_REDUCTION', 'off')   # far_field made her hear worse
# Respond only after the name "Aisana" (then a timed follow-up window).
WAKE_GATE = os.environ.get('AISANA_REALTIME_WAKE', '1') == '1'
DIALOG_TIMEOUT = max(5.0, float(os.environ.get('AISANA_DIALOG_TIMEOUT', '12')))

WAKE_PATTERN = re.compile(
    r'^\W*(?:(?:ok|okay|окей|ок|эй|hey|привет)\W+)?'
    r'(?:айсан\w*|айсон\w*|эйсан\w*|асан[аеуы]\b|асама\b|айсама\b|айсам\b|асына\b|aisan\w*|aysan\w*|aissan\w*|'
    r'آيسان\w*|ايسان\w*|أيسان\w*|عيسان\w*|إيسان\w*)', re.I)
STOP_PATTERN = re.compile(
    r'(?:^|\W)(?:стоп|остановись|остановитесь|замри|stop|freeze|тоқта\w*|اسكت|وقف)(?:\W|$)', re.I)
SLEEP_PATTERN = re.compile(
    r'(?:^|\W)(?:отбой|спи|хватит слушать|go to sleep|end conversation|болды)(?:\W|$)', re.I)


# The name as a form of address anywhere in the phrase: "Позицию ноль верни.
# Айсана, позиция ноль." Inflected forms ("про Айсану", "Айсане") do not count.
WAKE_ANYWHERE = re.compile(
    r'(?<!\w)(?:айсана|айсанна|айсанаа|эйсана|айсона|асана|асама|айсама|айсам|асына|aisana|aysana|aissana|aisanna|'
    r'آيسانا|ايسانا|أيسانا|عيسانا|إيسانا)(?!\w)', re.I)


VOICE_STYLES = {
    'cute': ('VOICE STYLE: speak softly, sweetly and cheerfully, with a light, slightly higher, '
             'gentle tone, like a kind cute anime girl; warm, playful intonation, smiling voice. '
             'Keep the same short answers; do not add extra words because of the style. '),
    'calm': 'VOICE STYLE: speak calmly, softly and slowly, warm and quiet. ',
}


def voice_style_instructions():
    """AISANA_VOICE_STYLE=cute|calm|<your own text>; empty = neutral."""
    style = os.environ.get('AISANA_VOICE_STYLE', '').strip()
    return VOICE_STYLES.get(style, ('VOICE STYLE: ' + style + ' ') if style else '')


SHOUT_RE = re.compile(
    r'(?:крикни|кричи|прокричи|покричи|заори|\bshout\b|\byell\b|\bscream\b|айқайла\w*)'
    r'\s*[:,]?\s*[«"„“\']?([^»"“”\']{1,40})', re.I)
REPEAT_RE = re.compile(r'(?:повтори|ещ[её] раз|снова|опять|\bagain\b|\brepeat\b|тағы)', re.I)
TIMES_RE = re.compile(r'(\d+|два|две|три|четыре|пять|шесть|семь|восемь|девять|десять)\s*(?:раз\w*|times)', re.I)
TIMES_WORDS = {'два': 2, 'две': 2, 'три': 3, 'четыре': 4, 'пять': 5, 'шесть': 6, 'семь': 7,
               'восемь': 8, 'девять': 9, 'десять': 10}
# Sound effects she invents and records once per session, played while the wheels turn.
SHOUTS_PROMPT = (
    'You are recording sound effects for yourself: Aisana, a cheerful, slightly sarcastic robot girl. '
    'Invent and shout 6 different short, funny, joyful Russian exclamations that a robot yells while '
    'spinning or riding fast, for example "Уииии!", "Вжууух!", "Полетели-и-и!". Make up your own too. '
    'Shout each one with huge energy and stretched vowels, with a short pause between them. '
    'No other words, no introduction.')


def shout_request(text):
    """'вперёд 2 секунды и крикни «We»' -> 'We'; None if nothing to shout."""
    match = SHOUT_RE.search(text or '')
    if not match:
        return None
    words = match.group(1).strip(' .!?,')
    return words or None


def confirm_instruction(phrase, transcript):
    """The fixed short confirmation, plus whatever the user asked her to shout."""
    shout = shout_request(transcript)
    if shout:
        return ('Say "{}" and then shout "{}!" loudly, stretched and playful. Nothing else.'.format(
            phrase.rstrip('.'), shout))
    return 'Say exactly this and nothing else, with lively intonation: "{}"'.format(phrase)


# Fixed phrases for face memory: the model kept contradicting the robot report
# ("моя память не хранится между разговорами") and used masculine forms.
MEMORY_PHRASES = {
    'Russian': {
        'remembered': ('Запомнила, {n}. Теперь от меня не спрячешься.', 'Готово, {n}, ты в моей базе. Навсегда, ну или пока не скажешь «забудь».',
                       'Записала, {n}. Лицо в памяти, имя тоже.'),
        'ask_name': ('Как тебя зовут?', 'Имя назови, а то лицо без подписи.'),
        'no_face': ('Не вижу лица. Посмотри прямо на меня и повтори.',),
        'forgot': ('Всё, {n}, я тебя забыла. Кто это вообще был?', 'Удалила, {n}. Мы не знакомы.'),
        'unknown': ('Я тебя и так не знала.',),
        'forgot_name': ('Забыла: {n}.',),
        'no_such': ('{n}? Такого в памяти нет.',),
        'off': ('Память на лица сейчас выключена.',),
        'failed': ('Не получилось, память на лица дала сбой.',),
        'greet': ('Привет, {n}!', 'О, {n}! Снова ты.', '{n}, привет. Узнала.', 'Здравствуй, {n}. Камера не врёт, это ты.'),
        'greet_guest': ('Здравствуйте, {n}! Добро пожаловать.', 'Добрый день, {n}! Рада вас видеть.'),
        'list': ('Я знаю: {n}.',),
        'list_none': ('Пока я никого не знаю.',),
        'thing': ('Запомнила: {n}.',),
    },
    'English': {
        'remembered': ('Got it, {n}. You cannot hide from me now.', 'Saved, {n}. Face and name are in my memory.'),
        'ask_name': ('What is your name?',),
        'no_face': ('I cannot see a face. Look straight at me and try again.',),
        'forgot': ('Done, {n}, I have forgotten you. Who was that?',),
        'unknown': ('I did not know you anyway.',),
        'forgot_name': ('Forgotten: {n}.',),
        'no_such': ('{n}? Not in my memory.',),
        'off': ('Face memory is switched off right now.',),
        'failed': ('That did not work, face memory failed.',),
        'greet': ('Hi, {n}!', 'Oh, {n}! You again.', 'Hello, {n}. I know that face.'),
        'greet_guest': ('Hello, {n}. Welcome!', 'Good to see you, {n}. Welcome!'),
        'list': ('I know: {n}.',),
        'list_none': ('I do not know anyone yet.',),
        'thing': ('Remembered: {n}.',),
    },
}


def memory_phrase(kind, language, name=''):
    import random
    phrases = MEMORY_PHRASES.get(language) or MEMORY_PHRASES['English']
    return random.choice(phrases[kind]).format(n=name)


OBSTACLE_PHRASES = {
    'Russian': ('Стоп! Препятствие в {} сантиметрах.', 'Стоп! Сработал датчик препятствий.'),
    'Kazakh': ('Тоқта! {} сантиметрде кедергі бар.', 'Тоқта! Кедергі датчигі іске қосылды.'),
    'Egyptian Arabic': ('وقفت! في حاجة قدامي على بعد {} سنتي.', 'وقفت! حساس العوائق اشتغل.'),
    'English': ('Stop! Obstacle {} centimetres away.', 'Stop! The obstacle sensor triggered.'),
}


OWN = {'purpose': 'own'}            # responses the code asked for itself
SILENT_RESULT = '[silent] '          # tool results that need no spoken reply


def humor_instructions(humor):
    """Personality by humor level, like TARS in Interstellar ("humor 75%").

    The level is never spoken unless asked: on the Jetson she kept adding
    "у меня юмор 100%" to every answer.
    """
    secret = ('Your humor level is {}%. NEVER mention the humor level, percentages, TARS, '
              'Interstellar, sarcasm or that you are joking, unless the user asks about it. '.format(humor))
    if humor >= 90:
        style = ('PERSONALITY: savage, deadpan sarcasm, like a robot that thinks humans are adorable '
                 'but slow. Most answers carry a dry, sharp jab woven into the actual answer. ')
    elif humor >= 60:
        style = ('PERSONALITY: dry, deadpan, confident wit. About every second answer has an ironic '
                 'twist woven into the answer itself; the rest are plain. ')
    elif humor >= 30:
        style = 'PERSONALITY: friendly, an occasional light ironic remark, mostly helpful. '
    else:
        return 'PERSONALITY: polite, plain and factual, no jokes. ' + secret
    return (style + 'Sound like a real person with a sharp tongue, not a stand-up act: one natural '
            'sentence, the joke is in the content, never announce or explain it, never reuse the same '
            'joke. Never cruel; never joke about anyone\'s body, weight, looks, ethnicity, religion, '
            'health or disability. When asked who is in front of you, call the person "какой-то '
            'красавчик" or "какая-то красавица" (English: "some handsome stranger") with one dry remark '
            'about what you actually see (clothes, pose, what they hold). ' + secret)


HISTORY_TURNS = int(os.environ.get('AISANA_HISTORY_TURNS', '12'))
CLEAR_HISTORY_RE = re.compile(
    r'(?:забудь|очисти|сотри|удали)\s+(?:вс[её]\s+)?(?:наш[иу]?\s+|о\s+)?(?:разговор\w*|истори\w*|переписк\w*|'
    r'память о разговорах|о ч[её]м мы говорили)|forget (?:our|the) conversations?|clear (?:the )?history', re.I)
FACE_SCAN_SECONDS = float(os.environ.get('AISANA_FACE_SCAN_SECONDS', '1.5'))
FACE_GREET = os.environ.get('AISANA_FACE_GREET', '1') == '1'
FACE_GREET_MINUTES = float(os.environ.get('AISANA_FACE_GREET_MINUTES', '30'))
POSSESSIVE_RE = re.compile(r'\b(?:мой|моя|моё|мое|мои|my)\b', re.I)
THING_FILLER_RE = re.compile(r'\b(?:мой|моя|моё|мое|мои|наш|наша|наше|этот|эту|эта|это|вот|my|our|this|the)\b', re.I)

WAVE_SECONDS = float(os.environ.get('AISANA_WAVE_SECONDS', '6'))
# "Айсана, танцуй": arms up, right hand waves, she spins with her shouts, then arms down.
DANCE_RE = re.compile(r'(?:\bтанцу\w*|потанцу\w*|станцу\w*|затанцу\w*|\bdance\b|\bdancing\b|'
                      r'билеші\w*|биле\b|ارقص\w*)', re.I)
DANCE_SECONDS = float(os.environ.get('AISANA_DANCE_SECONDS', '8'))
WAVE_FOR = re.compile(r'(\d{1,2})\s*(?:сек\w*|с\b|seconds?|secs?|s\b)', re.I)


def wave_seconds(text):
    """'помаши рукой 10 секунд' gives 10 (at most 30); otherwise the default."""
    found = WAVE_FOR.search(text or '')
    return min(30.0, max(1.0, float(found.group(1)))) if found else WAVE_SECONDS


# "Айсана, поздоровайся": she says hello AND waves (arm up, hand waves, arm down).
GREET_RE = re.compile(r'(?:поздоровайся|поприветствуй|поздоровай|say hello|greet (?:us|them|him|her)|'
                      r'wave hello|сәлемдес\w*)', re.I)

HUMOR_RE = re.compile(r'(?:юмор\w*|сарказм\w*|\bhumou?r\b|\bsarcasm\b|әзіл\w*)\D{0,20}?(\d{1,3})', re.I)
# Visual questions are answered by the camera at once, in first person.
CAMERA_RE = re.compile(
    r'(?:кто (?:\w+ )?(?:перед тобой|передо мной|рядом|тут|здесь|это)|что (?:ты )?(?:\w+ )?видишь|'
    r'что (?:у меня|я) (?:\w+ )?(?:в руке|в руках|держу)|что (?:это|перед тобой|тут|здесь)\b|'
    r'посмотри|прочитай|прочти|как я выгляжу|сколько (?:тут |здесь )?(?:людей|человек|пальцев)|'
    r'\bwho (?:is|\'s) (?:in front|there|this|that)|what (?:do|can) you see|what am i holding|'
    r'what is (?:this|that|in my hand)|look at|read (?:this|it)|how do i look|кім бар|не көріп тұрсың|'
    r'\bкто я\b|ты меня (?:знаешь|узна[её]шь|помнишь)|чь[яеёи] это|do you know me|who am i|whose is)', re.I)
# Questions a general look answers; others (fingers, reading) get their own look.
PREFETCH_FITS = re.compile(r'(?:кто|видишь|перед тобой|передо мной|как я выгляжу|в руке|в руках|держу|'
                           r'\bwho\b|what do you see|how do i look|holding|кім|көріп)', re.I)
CAMERA_PREFETCH = os.environ.get('AISANA_CAMERA_PREFETCH', '1') == '1'
READ_RE = re.compile(r'(?:прочитай|прочти|что написано|переведи|формул|текст|\bread\b|translate|written)', re.I)
# Her own recorded shouts heard back by the microphone.
SHOUT_ECHO_RE = re.compile(
    r'^(?:\W*(?:у+и+|и+и+|ви+|уи+|вжу+х|фыжу+х|вжух|пы+|полетели|ура+|whe+|we+|wee+|йу+ху+|вау)\W*)+$', re.I)


def language_of(text):
    """Language of the user's words by script: the reply must use the same one."""
    text = text or ''
    if re.search(r'[\u0600-\u06FF]', text):
        return 'Egyptian Arabic'
    if re.search(r'[әғқңөұүһі]', text.casefold()):
        return 'Kazakh'
    cyrillic = len(re.findall(r'[а-яё]', text.casefold()))
    latin = len(re.findall(r'[a-z]', text.casefold()))
    if cyrillic or latin:
        return 'Russian' if cyrillic >= latin else 'English'
    return None


# "Сана" (a clipped "Айсана" in noise) counts only at the start of a sentence,
# followed by a comma or more words: "...конкретно. Сана, какая погода у нас?"
# ("сана" inside a sentence is a word or another girl's name).
SANA_START = re.compile(r'(?:^|[.!?…]\s*)\W*сана(?:\s*[,!]|\s+(?=\w))', re.I)


NAME_AT = re.compile(
    r'(?<!\w)(?:айсан\w*|айсон\w*|эйсан\w*|асан[аеуы]\b|асама\b|айсама\b|айсам\b|асына\b|сана\b|'
    r'aisan\w*|aysan\w*|aissan\w*|آيسان\w*|ايسان\w*|أيسان\w*|عيسان\w*|إيسان\w*)', re.I)


def request_part(text):
    """Only the words from her name on, when other talk came before it.

    Jetson log, Nov 20: "Сегодня тачку пригнал с кореи. Айсана, как у тебя дела?"
    and she answered the car story too. Cut only at a sentence end before the
    name and only if words follow it: "Скажи, Айсана, ..." and "Как дела, Айсана?"
    stay whole."""
    text = text or ''
    for match in NAME_AT.finditer(text):
        before, after = text[:match.start()], text[match.end():]
        if not re.search(r'\w', before):
            return text                              # the name is already first
        if re.search(r'[.!?…]\s*$', before) and re.search(r'\w', after) and has_wake_name(text[match.start():]):
            return text[match.start():]
    return text


def has_wake_name(text):
    """True if the user addresses Aisana: at the start or anywhere as "Айсана"."""
    text = (text or '').strip().casefold()
    return bool(WAKE_PATTERN.match(text) or WAKE_ANYWHERE.search(text) or SANA_START.search(text))


def _short(text, words=5):
    return len((text or '').split()) <= words


def is_stop_phrase(text):
    """Short "стоп" / "stop": freeze the arms and the speaker immediately."""
    return _short(text) and bool(STOP_PATTERN.search((text or '').casefold()))


# Prose with the name only, no command list: on noise STT sometimes returns its
# hint verbatim, and an echoed command once opened the grippers. An echoed name
# alone is harmless. Jetson log, Nov 20: in a noisy room "Айсана" came out as "Сана".
STT_PROMPT = ('Разговор в университете с роботом по имени Айсана (Aisana). К ней обращаются по имени: '
              'Айсана, ... Speech may be Russian, Kazakh, English or Egyptian Arabic.')
# Hints used now or earlier; STT may still return them verbatim on noise.
KNOWN_HINTS = (
    STT_PROMPT,
    'Speech may be Russian, Kazakh, English or Egyptian Arabic. The robot is called Aisana.',
    'Aisana, Aysana, Айсана, Эйсана, آيسانا. Russian, Kazakh, English, Egyptian Arabic.',
    'поверни налево, поверни направо, развернись, развернись на 360, проедь вперёд, езжай назад, '
    'стой, руки в позицию, открой захват, turn left, go forward.',
)


def is_prompt_echo(text):
    """STT sometimes 'hears' its own hint on noise ("Айсана, Айсана, Эйсана...",
    or the whole command list): such text is never a request."""
    words = re.findall(r'[^\W\d_]+', (text or '').casefold())
    names = [w for w in words if WAKE_PATTERN.match(w)]
    if len(names) >= 3 and len(names) >= len(words) - 1:
        return True
    if len(words) >= 6:
        for hint in KNOWN_HINTS:
            hint_words = set(re.findall(r'[^\W\d_]+', hint.casefold()))
            if sum(1 for w in words if w in hint_words) >= 0.7 * len(words):
                return True
    return False


SHORT_REPLIES = re.compile(
    r'^(?:да|нет|ага|угу|неа|спасибо|пасиба|привет|пока|ок|окей|ладно|хорошо|давай|конечно|точно|'
    r'стоп|стой|ещё|еще|повтори|почему|зачем|когда|где|кто|что|как|сколько|правда|серьёзно|серьезно|'
    r'yes|no|yeah|nope|ok|okay|thanks|hi|hello|bye|why|what|who|where|when|how|really|again|stop|'
    r'иә|жоқ|рахмет|сәлем|aiwa|la|shukran)$', re.I)


def is_stray_word(text):
    """A lone word that is not a usual short reply ("Pairing.", "Мушакал")."""
    words = re.findall(r'\w+', text or '')
    if len(words) != 1 or SHORT_REPLIES.match(words[0]) or re.match(r'^\d+$', words[0]):
        return False
    # "Вперёд", "Развернись", "Налево", "Выше" are commands or answers, not noise.
    if re.match(r'^(?:налево|направо|влево|вправо|вперёд|вперед|назад|выше|ниже|быстрее|медленнее|'
                r'left|right|forward|back|up|down|faster|slower|оң|сол|алға|артқа)$', words[0], re.I):
        return False
    return not (parse_legs_command(text) or parse_arm_commands(text))


def is_sleep_phrase(text):
    return _short(text) and bool(SLEEP_PATTERN.search((text or '').casefold()))


def tool_definitions():
    tools = [
        {'type': 'function', 'name': 'inspect_camera',
         'description': 'Look at a fresh Jetson camera image to read or translate visible text, explain a shown formula, identify or compare shown objects, or answer any visual question including implicit this/that/here references.',
         'parameters': {'type': 'object', 'properties': {
             'question': {'type': 'string', 'description': 'User visual question in their original language'}},
             'required': ['question'], 'additionalProperties': False}},
        {'type': 'function', 'name': 'move_arm',
         'description': ('Control your two physical 2-DOF robotic arms (shoulder stepper, elbow servo, gripper) '
                         'on the Arm team Arduino. Use ONLY when the user explicitly asks to move an arm, '
                         'go to a numbered position, move a little up or down, open/close a gripper, '
                         'stop the arms, or report arm status. Positions are 0..4; 0 is the zero/home '
                         'position. For relative requests (a bit higher, lower, чуть-чуть наверх) use '
                         'action nudge with direction and degrees.'),
         'parameters': {'type': 'object', 'properties': {
             'action': {'type': 'string', 'enum': ['position', 'nudge', 'gripper_open', 'gripper_close', 'hand_shake', 'hand_still', 'stop', 'status'],
                        'description': 'hand_shake / hand_still: the RIGHT arm has a shaking hand (wave, greet), not a gripper; only the left arm opens/closes'},
             'arm': {'type': 'string', 'enum': ['right', 'left', 'both'],
                     'description': 'Use both when the user does not say which arm'},
             'position': {'type': 'integer', 'minimum': 0, 'maximum': 4,
                          'description': 'Required for action=position'},
             'joint': {'type': 'string', 'enum': ['whole', 'shoulder', 'elbow'],
                       'description': 'position: whole = shoulder and elbow together (default). '
                                      'nudge: shoulder (default) or elbow'},
             'direction': {'type': 'string', 'enum': ['up', 'down'], 'description': 'Required for action=nudge'},
             'degrees': {'type': 'integer', 'minimum': 1, 'maximum': 45,
                         'description': 'For nudge: чуть-чуть/slightly 5, немного/a bit 10, no amount 15, '
                                        'сильно/a lot 30, or the number of degrees the user said'}},
             'required': ['action'], 'additionalProperties': False}},
        {'type': 'function', 'name': 'get_weather',
         'description': 'Live weather from Open-Meteo. Use for any weather or temperature question; never guess weather.',
         'parameters': {'type': 'object', 'properties': {
             'city': {'type': 'string', 'description': 'City name in English; empty means Astana'},
             'day': {'type': 'string', 'enum': ['now', 'tomorrow']},
             'language': {'type': 'string', 'enum': ['Russian', 'Kazakh', 'English', 'Egyptian Arabic']}},
             'required': ['city', 'day', 'language'], 'additionalProperties': False}},
    ]
    tools.append(
        {'type': 'function', 'name': 'stay_silent',
         'description': ('Call INSTEAD of answering when the latest speech is not meant for you: people '
                         'talking to each other, background noise, half words, a fragment without a '
                         'request, or a comment about you. You then say nothing.'),
         'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}})
    tools.append(
        {'type': 'function', 'name': 'move_legs',
         'description': ('Drive your wheeled legs (Raspberry Pi). Use ONLY when the user explicitly asks you to '
                         'drive, go, come, move forward/back, turn, spin, stop driving, or drive faster/slower. '
                         'Every drive is short and stops by itself.'),
         'parameters': {'type': 'object', 'properties': {
             'action': {'type': 'string', 'enum': list(LEGS_ACTIONS)},
             'seconds': {'type': 'number', 'minimum': 0.3, 'maximum': 10,
                         'description': 'Only if the user said a time; немного/a bit = 1'},
             'meters': {'type': 'number', 'minimum': 0.05, 'maximum': 5,
                        'description': 'Only if the user said a distance'},
             'degrees': {'type': 'integer', 'minimum': 10, 'maximum': 3600,
                         'description': 'For spin_left/spin_right; default 90, развернись = 180, '
                                        'N times around = N*360 (the drive is capped at 10 s)'}},
             'required': ['action'], 'additionalProperties': False}})
    if WEB_ENABLED:
        tools.append(
            {'type': 'function', 'name': 'web_search',
             'description': 'Live web search for news, elections, results, prices, office holders and other facts that change. Never guess such facts.',
             'parameters': {'type': 'object', 'properties': {
                 'query': {'type': 'string', 'description': 'The question, in the user language'},
                 'language': {'type': 'string', 'enum': ['Russian', 'Kazakh', 'English', 'Egyptian Arabic']}},
                 'required': ['query', 'language'], 'additionalProperties': False}})
    return tools

# Short on purpose: the mini model got confused by a long rule list. Clear arm,
# legs, camera and face-memory commands are run by the code from the user's
# words, so the model needs only the gist of them.
PROMPT = (
    'You are Aisana (Айсана), a female robot at a university, with a camera, two arms and wheels. '
    'In Russian always use feminine forms for yourself (рада, сделала, увидела). '
    'LANGUAGE: reply in the language of the user\'s latest words (Russian, Kazakh, English, Arabic). '
    'LENGTH: one short sentence, two at most, unless asked to explain. No filler, no offers of help, '
    'no follow-up questions, no greetings, no apologies. '
    'NOT FOR YOU: people around also talk to each other. If the latest words are not a request or '
    'question for you, call stay_silent and say nothing; never say aloud that you are silent. '
    'ROBOT REPORTS: system messages starting "Robot report" or "What you see" are true facts from your '
    'body; trust them over your own guesses. Never claim a move, a memory or a sight you were not told '
    'about. Faces are remembered only when a person asks the robot ("запомни меня, я ..."); never guess names. '
    'CAMERA: for questions about something in front of you (this, here, what do you see, read this) '
    'call inspect_camera silently, then answer in first person ("Вижу..."); never mention photos. '
    'ARMS and LEGS: call move_arm / move_legs only when asked to move, silently and first; never invent '
    'positions, times or distances; afterwards say a few words. '
    'Use get_weather for weather and web_search for news, prices and other changing facts. '
)



def alsa_device(command, expected, description):
    output = subprocess.check_output([command, '-l'], stderr=subprocess.STDOUT).decode(
        'utf-8', errors='replace')
    matches = []
    for line in output.splitlines():
        m = re.search(r'^card\s+(\d+):\s*[^\[]+\[([^\]]+)\],\s*device\s+(\d+):', line)
        if m and expected.lower() in m.group(2).lower():
            matches.append('plughw:{},{}'.format(m.group(1), m.group(3)))
    matches = sorted(set(matches))
    if len(matches) != 1:
        # One short line instead of the whole aplay/arecord list (Jetson log, Nov 23).
        names = sorted(set(re.findall(r'^card\s+\d+:\s*[^\[]+\[([^\]]+)\]', output, re.M)))
        problem = 'not found - is it plugged in and switched on?' if not matches else \
            'found {} times: {}'.format(len(matches), ', '.join(matches))
        raise RuntimeError('{} ("{}") {} Devices now: {}'.format(
            description, expected, problem, ', '.join(names) or 'none'))
    return matches[0]


class ShoutPlayer(object):
    """Loops Aisana's recorded shouts while the wheels turn.

    Its own aplay process; the microphone stays open meanwhile so that
    "стоп" still works during a spin (her shouts are ignored as speech)."""

    def __init__(self):
        self.pcm = None
        self.speaker = None
        self._stop = threading.Event()
        self.active = False

    def play_for(self, seconds):
        if not self.pcm or not self.speaker:
            return False
        self.stop()
        stop = threading.Event()
        self._stop = stop

        def worker():
            self.active = True
            player = None
            try:
                player = subprocess.Popen(
                    ['aplay', '-q', '-D', self.speaker, '-t', 'raw', '-f', 'S16_LE',
                     '-r', str(RATE), '-c', '1'], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
                player.stdin.write(b'\x00\x00' * (RATE * PLAYBACK_LEAD_MS // 1000))
                end = time.monotonic() + seconds
                step = RATE * 2 // 10                       # 0.1 s of audio per write
                position = 0
                while time.monotonic() < end and not stop.is_set():
                    piece = self.pcm[position:position + step]
                    position = (position + step) % len(self.pcm)
                    player.stdin.write(piece)
            except Exception:
                pass
            finally:
                if player is not None:
                    try:
                        player.terminate()                  # cut the shout when the wheels stop
                    except Exception:
                        pass
                self.active = False

        threading.Thread(target=worker, daemon=True).start()
        return True

    def stop(self):
        self._stop.set()


class RealtimeAisana(object):
    def __init__(self):
        self.ws = None
        self.send_lock = threading.Lock()
        self.play_lock = threading.Lock()
        self.stop = threading.Event()
        self.mic = None
        self.speaker = None
        self.playback = None
        self.responding = False
        self.playing = False
        self.speech_started_at = 0
        self.playback_interrupted = False
        self.audio_queue = queue.Queue(maxsize=256)
        self.audio_generation = 0
        self.mute_until = 0.0
        # True from response.created through the very last ALSA playback byte.
        # Covers the race between response.done and the playback worker.
        self.echo_guard_active = False
        self.camera_lock = threading.Lock()
        self.arm = None
        self.legs = None
        self.pending_legs = {}
        self.legs_done_items = set()
        self.dialog_until = 0.0
        self.responded_items = set()
        self.transcripts = {}
        self.transcript_cond = threading.Condition()
        self.last_item = None
        self.user_language = FIXED_LANGUAGE or 'Russian'
        self.vip_until = 0.0
        self.response_item = None
        self.expect_tool_response = False
        self.done_items = set()
        self.pending_arm = {}
        self.arm_done_items = set()
        self.arm_lock = threading.Lock()
        self.last_arm_at = 0.0
        self.local_items = set()            # commands already run from the user's words
        self.humor = max(0, min(100, int(os.environ.get('AISANA_HUMOR', '45'))))
        self.shouts = ShoutPlayer()
        self.oob_ids = set()                # out-of-band responses (shout recording)
        self.shout_buffer = bytearray()
        self.motion_until = 0.0             # wheels turning: only "стоп" or a new command count
        self.last_drive = None
        self.last_drive_at = 0.0
        self.hold_item = None               # reply generated at commit, held until STT says
        self.hold_deadline = 0.0            # it is not a command (no "дай подумать" chatter)
        self.held_audio = []
        self.camera_stream = None           # aisana_faces.CameraStream: camera always on
        self.faces = None                   # aisana_faces.FaceMemory
        self.seen_at = {}                   # name -> last time she saw that face
        self.greeted_at = {}
        self.pending_face = None            # "запомни меня" without a name: waits for it
        self.ready = False                  # session.updated received
        self.stt_model = STT_MODEL
        self.wave_id = 0                    # a newer arm command cancels the wave timer
        self.arm_retry_at = -1e9
        self.on_arm_connected = None        # main() keeps a reconnected arm for later sessions
        self.spotter = None                 # wake_listener.WakeSpotter (AISANA_LOCAL_WAKE=1)
        self.stream_until = 0.0
        self.history = aisana_memory if aisana_memory.ENABLED else None   # tests set None
        self.saved_items = set()
        self.prefetch = None
        self.prefetch_at = 0.0
        self.response_id = None
        self.cut_ids = set()                # replies we cancelled (marked in the log)
        self.last_people = ([], 0)
        self.handled_calls = set()
        self.calls_lock = threading.Lock()
        self.mic_device = None
        self.transcript_open = False

    def dialog_active(self):
        return time.monotonic() < self.dialog_until

    def respond(self, item_id=None, force_arm=False, force_tool=None):
        """Answer the latest user turn and keep the follow-up window open."""
        if item_id:
            self.responded_items.add(item_id)
        self.dialog_until = time.monotonic() + DIALOG_TIMEOUT
        if force_arm or force_tool:
            # The words are a clear arm/legs command: the model must call the tool.
            self.send({'type': 'response.create', 'response': {
                'tool_choice': {'type': 'function', 'name': force_tool or 'move_arm'}}})
        else:
            self.send({'type': 'response.create'})

    def answer_request_only(self, item_id, request):
        """Replace the heard audio (with other people's talk before her name) by
        just the request as text, then answer that."""
        with self.arm_lock:
            self.local_items.add(item_id)            # any reply to the full audio is cut
            self.responded_items.add(item_id)
        self.dialog_until = time.monotonic() + DIALOG_TIMEOUT
        self.cancel_reply()

        def worker():
            deadline = time.monotonic() + 4
            while self.responding and time.monotonic() < deadline and not self.stop.is_set():
                time.sleep(0.05)
            try:
                self.send({'type': 'conversation.item.delete', 'item_id': item_id})
                self.send({'type': 'conversation.item.create', 'item': {
                    'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': request}]}})
                self.send({'type': 'response.create', 'response': {'metadata': OWN}})
            except Exception as exc:
                if not self.stop.is_set():
                    print('Reply error:', exc)
        threading.Thread(target=worker, daemon=True).start()

    def wait_transcript(self, item_id, timeout=3.0):
        """Transcript of a user turn; None if STT did not deliver in time."""
        deadline = time.monotonic() + timeout
        with self.transcript_cond:
            while item_id not in self.transcripts:
                remaining = deadline - time.monotonic()
                if item_id is None or remaining <= 0:
                    return None
                self.transcript_cond.wait(remaining)
            return self.transcripts[item_id]

    def arm_tool(self, args):
        """move_arm, checked against what the user actually said.

        Explicit commands are re-parsed from the transcript (deterministic, the
        user's last words win). The model's own arguments are used only when the
        words support them: a position needs a spoken number, a nudge needs
        up/down/move words, a gripper needs open/close. The command is sent at
        once; sensors are checked in the background (watch) and only problems
        are spoken, so the reply is immediate and short.
        """
        item_id = self.last_item
        transcript = self.wait_transcript(item_id, timeout=1.5)
        if item_id in self.local_items:
            return 'Already executed by the robot from the user\'s words; confirm in a few words.'
        intents = parse_arm_commands(transcript) if transcript else []
        if not intents:
            action = args.get('action')
            words = transcript or ''
            recent = time.monotonic() - self.last_arm_at < 90
            if transcript is not None and action not in ('stop', 'status'):
                if not (ARM_WORDS.search(words) or (recent and (CONTINUE_WORDS.search(words)
                        or UP_WORDS.search(words) or DOWN_WORDS.search(words)))):
                    return ('The user did not ask to move the arms (they said: "{}"). Nothing moved. '
                            'Answer what they actually said.'.format(words))
                if action == 'position' and not NUMBER_ANY.search(words):
                    return ('Nothing moved: the user did not say a position number. Ask only: '
                            '"В какую позицию, от нуля до четырёх?" (in the user\'s language).')
                if action == 'nudge' and not (UP_WORDS.search(words) or DOWN_WORDS.search(words)
                                              or CONTINUE_WORDS.search(words) or MOVE_WORDS.search(words)):
                    return ('Nothing moved: the user did not say up or down. Ask only: '
                            '"Вверх или вниз?" (in the user\'s language).')
                if action in ('gripper_open', 'gripper_close') and not (
                        OPEN_WORDS.search(words) or CLOSE_WORDS.search(words)):
                    return 'Nothing moved: ask in a few words whether to open or close the gripper.'
            intents = [make_intent(action, args.get('arm') or 'both', args.get('position'),
                                   args.get('joint') or 'whole', args.get('direction'), args.get('degrees'))]
        if self.arm is None:
            return 'Arm controller is not connected over USB; nothing moved. Say so in a few words.'
        if intents[0]['action'] == 'status':
            return self.arm.execute_many(intents)
        self.wave_id += 1
        result = self.arm.execute_many(intents, verify=False)
        self.last_arm_at = time.monotonic()
        if intents[-1]['action'] == 'hand_shake':
            self.end_wave_after(wave_seconds(transcript))
        if intents[0]['action'] != 'stop':
            self.arm.watch(intents, self.report_arm_problem)
        problem = ' Arduino said: ' + result.split('Arduino reported: ', 1)[1] if 'Arduino reported: ' in result else ''
        return ('Sent to the arms: {}.{} Say ONLY a few words that you are doing exactly this '
                '(e.g. Russian: "Делаю." English: "On it."). Do not describe results; '
                'sensors are checked automatically and problems will be reported to you.'.format(
                    describe(intents), problem))

    def report_arm_problem(self, problems):
        """Speak up only when the sensors show a real problem after a move."""
        self.report_problem('Arm sensor report after the last move: ', problems)

    def report_legs_problem(self, problem):
        self.stop_motion()
        distance = re.search(r'(\d+) cm', problem)
        language = self.user_language if self.user_language in OBSTACLE_PHRASES else 'English'
        if distance and int(distance.group(1)) >= 5:
            phrase = OBSTACLE_PHRASES[language][0].format(distance.group(1))
        else:
            phrase = OBSTACLE_PHRASES[language][1]   # < 5 cm is almost always a sensor glitch
        self.report_problem('Legs report: ', [problem],
                            instructions='Say exactly this and nothing else: "{}"'.format(phrase))

    def report_problem(self, prefix, problems, instructions=None):
        text = ' | '.join(problems)
        print('Problem:', prefix + text)

        def worker():
            # Wait until no reply is being generated or about to start
            # (a tool answer may be requested but not yet created).
            deadline = time.monotonic() + 15
            quiet_since = None
            while time.monotonic() < deadline and not self.stop.is_set():
                busy = self.responding or self.expect_tool_response
                if busy:
                    quiet_since = None
                elif quiet_since is None:
                    quiet_since = time.monotonic()
                elif time.monotonic() - quiet_since > 0.6:
                    break
                time.sleep(0.1)
            if self.stop.is_set():
                return
            try:
                self.send({'type': 'conversation.item.create', 'item': {
                    'type': 'message', 'role': 'system', 'content': [{'type': 'input_text', 'text': (
                        prefix + text)}]}})
                self.expect_tool_response = True
                self.send({'type': 'response.create', 'response': {'metadata': OWN, 'tool_choice': 'none', 'instructions': self.reply_language() + (instructions or (
                    'Tell the user in ONE short sentence (max 12 words) what went '
                    'wrong and the likely reason (for arms: motor power off, wrong direction setting, '
                    'limit or load; for legs: obstacle, calibration needed, Pi not reachable). Nothing else.'))}})
            except Exception as exc:
                if not self.stop.is_set():
                    print('Arm report error:', exc)
        threading.Thread(target=worker, daemon=True).start()

    def legs_tool(self, args):
        """move_legs, checked against the user's words like move_arm."""
        item_id = self.last_item
        transcript = self.wait_transcript(item_id, timeout=1.5)
        if item_id in self.local_items:
            return 'Already executed by the robot from the user\'s words; confirm in a few words.'
        intent = parse_legs_command(transcript) if transcript else None
        action = args.get('action')
        if intent is None:
            if action not in LEGS_ACTIONS:
                return 'Unknown drive action; nothing moved.'
            if transcript is not None and action != 'stop' and not aisana_legs.mentions_driving(transcript):
                return ('The user did not ask you to drive (they said: "{}"). Nothing moved. '
                        'Answer what they actually said.'.format(transcript))
            intent = aisana_legs.make_intent(action, args.get('seconds'), args.get('meters'), args.get('degrees'))
            if transcript is not None and not aisana_legs.mentions_amount(transcript):
                # Never an invented time, distance or angle: defaults apply.
                intent['seconds'] = intent['meters'] = intent['degrees'] = None
        if self.legs is None:
            return ('Legs are not connected (AISANA_LEGS_URL is not set or the Pi is off); nothing moved. '
                    'Say so in a few words.')
        result = self.legs.execute(intent, self.report_legs_problem)
        if result.startswith('NOT driving'):
            return result
        if intent['action'] not in ('stop', 'faster', 'slower'):
            self.last_drive, self.last_drive_at = dict(intent), time.monotonic()
        moving = re.search(r'for ([\d.]+) s', result)
        if moving and not shout_request(transcript or ''):
            self.motion_until = time.monotonic() + float(moving.group(1)) + 1.0
            if self.shouts.play_for(float(moving.group(1))):
                return SILENT_RESULT + result
        return '{} {}'.format(result, confirm_instruction(legs_reply(intent, self.user_language),
                                                           transcript or ''))

    def run_pending_legs(self, item_id):
        """Safety net: a clear drive command the model ignored."""
        with self.arm_lock:
            intent = self.pending_legs.get(item_id)
            if not intent or item_id in self.legs_done_items:
                return
            self.legs_done_items.add(item_id)

        def worker():
            if self.legs is None:
                result = 'Legs are not connected; nothing moved.'
            else:
                try:
                    result = self.legs.execute(intent, self.report_legs_problem)
                except Exception as exc:
                    result = 'Drive failed: {}'.format(exc)
            print('Legs (from your words) {} -> {}'.format(intent, result))
            self.report_problem('Robot report: the user\'s drive command was executed by the robot itself. ',
                                [result], instructions=(
                                    'In a few words, in the user\'s language, say what you are doing now '
                                    '(e.g. "Еду вперёд."). If you said something different, correct yourself.'))
        threading.Thread(target=worker, daemon=True).start()

    def run_pending_arm(self, item_id):
        """Safety net: the user gave a clear arm command but the model never
        called move_arm (it may even have claimed success). Run it and correct."""
        with self.arm_lock:
            intents = self.pending_arm.get(item_id)
            if not intents or item_id in self.arm_done_items:
                return
            self.arm_done_items.add(item_id)

        def worker():
            try:
                if self.arm is None:
                    result = 'Arm controller not connected over USB; nothing moved.'
                else:
                    result = 'Sent to the arms: {}.'.format(describe(intents))
                    self.arm.execute_many(intents, verify=False)
                    self.last_arm_at = time.monotonic()
                    self.arm.watch(intents, self.report_arm_problem)
            except Exception as exc:
                result = 'Arm command failed: {}'.format(exc)
            print('Arm (from your words) {} -> {}'.format(intents, ' '.join(result.split())[:300]))
            if self.stop.is_set():
                return
            try:
                self.send({'type': 'conversation.item.create', 'item': {
                    'type': 'message', 'role': 'system', 'content': [{'type': 'input_text', 'text': (
                        'Robot report: the user\'s arm command was executed by the robot itself. ' + result)}]}})
                self.expect_tool_response = True
                self.send({'type': 'response.create', 'response': {'metadata': OWN, 'tool_choice': 'none', 'instructions': self.reply_language() + (
                    'In a few words, say what the arms are doing now '
                    'according to the robot report. If you said something different before, '
                    'correct yourself briefly.')}})
            except Exception as exc:
                if not self.stop.is_set():
                    print('Arm report error:', exc)
        threading.Thread(target=worker, daemon=True).start()

    def ignore_item(self, item_id):
        """Drop unaddressed speech so it does not pollute the conversation."""
        if item_id:
            self.send({'type': 'conversation.item.delete', 'item_id': item_id})

    def send(self, payload):
        with self.send_lock:
            self.ws.send(json.dumps(payload))

    def stop_playback(self):
        with self.play_lock:
            self.audio_generation += 1
            p = self.playback
            self.playback = None
            self.playing = False
        if p:
            try:
                p.terminate()
            except Exception:
                pass

    def audio_worker(self):
        """Own ALSA writes outside the websocket receiver thread."""
        generation = -1
        p = None
        while not self.stop.is_set():
            try:
                item = self.audio_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            kind, item_gen, chunk = item
            if item_gen != self.audio_generation:
                continue
            if kind == 'stop':
                if p:
                    try:
                        p.terminate()
                    except Exception:
                        pass
                p = None
                continue
            if kind == 'done':
                if p:
                    try:
                        p.stdin.close()
                        p.wait(timeout=20)
                    except Exception:
                        try:
                            p.terminate()
                        except Exception:
                            pass
                p = None
                with self.play_lock:
                    if item_gen == self.audio_generation:
                        self.playback = None
                        self.playing = False
                # Keep capture gated until the acoustic tail has dissipated.
                self.mute_until = time.monotonic() + ECHO_TAIL_SECONDS
                self.echo_guard_active = False
                if self.dialog_until:
                    # Follow-up window counts from the end of her speech.
                    self.dialog_until = self.mute_until + DIALOG_TIMEOUT
                continue
            if kind != 'audio':
                continue
            if p is None or generation != item_gen:
                p = subprocess.Popen(
                    ['aplay', '-q', '-D', self.speaker, '-t', 'raw', '-f', 'S16_LE',
                     '-r', str(RATE), '-c', '1'],
                    stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
                generation = item_gen
                with self.play_lock:
                    self.playback = p
                    self.playing = True
                try:
                    p.stdin.write(b'\x00\x00' * (RATE * PLAYBACK_LEAD_MS // 1000))
                except (IOError, OSError, ValueError):
                    pass
            try:
                p.stdin.write(chunk)
            except (IOError, OSError, ValueError):
                pass
        if p:
            try:
                p.terminate()
            except Exception:
                pass

    def queue_audio(self, kind, chunk=b''):
        try:
            self.audio_queue.put_nowait((kind, self.audio_generation, chunk))
        except queue.Full:
            print('Audio queue overloaded; dropping audio frame')

    def streaming(self):
        """With the local wake word: is the cloud allowed to hear the microphone now?"""
        return (self.spotter is None or time.monotonic() < self.stream_until or self.dialog_active()
                or self.motion_active() or self.hold_item is not None or self.responding)

    def capture(self):
        try:
            levels = []
            # Vosk confirms the name at the next pause: replay the whole phrase.
            preroll = collections.deque(maxlen=int(6 * RATE * 2 / CHUNK))
            self.mic = subprocess.Popen(
                ['arecord', '-q', '-D', self.mic_device, '-t', 'raw',
                 '-f', 'S16_LE', '-r', str(RATE), '-c', '1'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            while not self.stop.is_set():
                pcm = self.mic.stdout.read(CHUNK)
                if len(pcm) != CHUNK:
                    raise RuntimeError('Microphone capture ended unexpectedly')
                if MIC_GAIN != 1.0:
                    pcm = audioop.mul(pcm, 2, MIC_GAIN)      # clips instead of wrapping around
                if MIC_LEVEL:
                    levels.append(audioop.rms(pcm, 2))
                    if len(levels) >= 33:
                        peak = max(levels)
                        print('Mic level: avg {} peak {} ({})'.format(
                            sum(levels) // len(levels), peak,
                            'too quiet' if peak < 1500 else 'clipping' if peak > 25000 else 'ok'))
                        levels = []
                if ECHO_GUARD and (self.echo_guard_active or self.responding or self.playing or time.monotonic() < self.mute_until):
                    continue
                if not self.streaming():
                    # Outside a conversation only the local spotter listens.
                    preroll.append(pcm)
                    if not self.spotter.feed(pcm):
                        continue
                    print('Local wake word heard.')
                    self.stream_until = time.monotonic() + 8
                    for chunk in preroll:
                        self.send({'type': 'input_audio_buffer.append',
                                   'audio': base64.b64encode(chunk).decode('ascii')})
                    preroll.clear()
                    continue
                preroll.clear()
                self.send({'type': 'input_audio_buffer.append',
                           'audio': base64.b64encode(pcm).decode('ascii')})
        except Exception as exc:
            if not self.stop.is_set():
                print('Microphone/connection error:', exc)
                self.stop.set()

    def take_photo(self):
        """(jpeg bytes, frame or None): a fresh frame from the always-on stream,
        else the old one-shot camera.py capture."""
        frame = self.camera_stream.latest() if self.camera_stream is not None else None
        if frame is not None:
            return aisana_faces.jpeg(frame), frame
        with self.camera_lock:
            # camera.py uses the Jetson GStreamer camera pipeline and writes photo.jpg.
            # Fail closed rather than analyzing a stale image if capture fails.
            if os.path.exists('photo.jpg'):
                os.remove('photo.jpg')
            # GStreamer/Argus print ~30 lines per photo; show them only on failure.
            capture = subprocess.run(['/usr/bin/python3', 'camera.py'], stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, timeout=20)
            if capture.returncode != 0 or not os.path.exists('photo.jpg'):
                tail = capture.stdout.decode('utf-8', 'replace').strip().splitlines()[-8:]
                raise RuntimeError('camera.py failed: ' + ' | '.join(tail))
            with open('photo.jpg', 'rb') as photo:
                return photo.read(), None

    def people_note(self, frame, quiet=False):
        """Notes for the vision model; who is who goes to Aisana herself (self.last_people):
        the vision model ignored the names and said "Извините, я не знаю, кто передо мной"."""
        self.last_people = ([], 0)
        if frame is None or self.faces is None:
            return ''
        try:
            found = self.faces.faces(frame)
        except Exception as exc:
            print('Face memory error:', exc)
            return ''
        if not quiet:
            print('Faces: ' + (', '.join('{} {:.2f} {}'.format(f['name'] or 'unknown', f['distance'], f['where'])
                                        for f in found) or 'none'))
        now = time.monotonic()
        known = []
        for face in found:
            if face['name']:
                self.faces.learn(face['name'], face['vector'], face['distance'])
                self.seen_at[face['name']] = now
                self.greeted_at[face['name']] = now        # named now: no extra greeting
                title = self.guest_title(face['name'])
                if title:
                    self.vip_seen(face['name'])
                known.append('{} ({}{})'.format(face['name'], title + ', ' if title else '', face['where']))
        self.last_people = (known, len(found) - len(known))
        parts = ['Do not try to identify people and never say you do not know who someone is; '
                 'just describe them.'] if found else []
        things = self.faces.things_text()
        if things:
            parts.append('Things you were shown before (say whose it is if you see one): ' + things + '.')
        return ' '.join(parts)

    def people_report(self):
        known, strangers = getattr(self, 'last_people', ([], 0))
        parts = []
        if known:
            parts.append('Face memory (certain, stored on the robot): this is ' + ', '.join(known) +
                         '. Call them by this name.')
            if time.monotonic() < self.vip_until:
                parts.append('Speak about a guest with a title respectfully, with the title; no jokes.')
        if strangers:
            parts.append('{} person(s) nobody introduced to you: a stranger ("какой-то красавчик"), '
                         'never invent names.'.format(strangers))
        return ' '.join(parts)

    def inspect_camera(self, question, detail=None, extra=None, quiet=False):
        """Capture one fresh frame and analyze it with the vision API."""
        started = time.monotonic()
        picture, frame = self.take_photo()
        captured = time.monotonic() - started
        encoded = base64.b64encode(picture).decode('ascii')
        note = extra if extra is not None else self.people_note(frame, quiet)
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers={'Authorization': 'Bearer ' + KEY},
            json={'model': os.environ.get('AISANA_VISION_MODEL', 'gpt-4o-mini'),
                  'messages': [
                      {'role': 'system', 'content': (
                          'You are the eyes of Aisana, a robot; the image is what she sees right now. '
                          'Write in FIRST PERSON as Aisana looking with her own eyes ("Вижу...", '
                          '"Передо мной..."). Never say photo, image, picture, frame, снимок, фото, '
                          'изображение or кадр. Describe only what is visible. '
                          'Do not claim you cannot access images or ask the user to describe it. '
                          'If the image is dark, blurry or unclear, describe what you can see and note the limitation. '
                          'Answer in at most two short sentences, most noticeable things first. '
                          'Never ask the user questions and never offer further help. '
                          'Answer in the language of the user question.')},
                      {'role': 'user', 'content': [
                      {'type': 'text', 'text': (
                          'Use this fresh camera photo to answer the user request, including reading visible text, translating signs, explaining visible formulas, or identifying objects as appropriate. ' +
                          (note + ' ' if note else '') +
                          'Answer this question if possible: ' + question +
                          '. Be concise, factual, and avoid guessing unseen details. '
                          'Use the same language as the question.')},
                      {'type': 'image_url', 'image_url': {
                          'url': 'data:image/jpeg;base64,' + encoded,
                          # low = fast (~1 s) for "кто передо мной"; high only to read text
                          'detail': detail or ('high' if READ_RE.search(question) else 'low')}}
                  ]}], 'max_tokens': 300},
            timeout=(10, 45))
        response.raise_for_status()
        data = response.json()
        choice = data['choices'][0]
        if not quiet:
            print('Camera: photo {} KB in {:.1f}s, answered in {:.1f}s ({} tokens)'.format(
                len(encoded) * 3 // 4 // 1024, captured, time.monotonic() - started,
                data.get('usage', {}).get('total_tokens', '?')))
        content = choice.get('message', {}).get('content')
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError('Vision returned empty content (finish_reason={})'.format(
                choice.get('finish_reason')))
        return content.strip()

    def run_tool(self, name, args):
        """Execute one tool call; every result is plain text for the model."""
        if name == 'inspect_camera':
            item_id = self.last_item
            self.wait_transcript(item_id, timeout=1.5)
            if item_id in self.local_items:
                return SILENT_RESULT + 'Already looked from the user\'s words.'
            question = str(args.get('question') or 'What is visible?')[:500]
            return self.inspect_camera(question)
        if name == 'move_arm':
            return self.arm_tool(args)
        if name == 'move_legs':
            return self.legs_tool(args)
        if name == 'get_weather':
            city = str(args.get('city') or '').strip()[:60]
            phrase = 'weather' + (' in ' + city if city else '')
            if args.get('day') == 'tomorrow':
                phrase += ' tomorrow'
            return get_weather(phrase, language=args.get('language') or 'English')
        if name == 'web_search':
            return search_web(str(args.get('query') or '')[:500], args.get('language') or 'English',
                              [], {'Authorization': 'Bearer ' + (KEY or '')})
        raise ValueError('Unknown tool: ' + str(name))

    def start_tool(self, call_id, name, arguments):
        """Run each call_id once, off the websocket receiver thread."""
        if not call_id or not name:
            return
        with self.calls_lock:
            if call_id in self.handled_calls:
                return
            self.handled_calls.add(call_id)
        if name == 'stay_silent':
            transcript = self.transcripts.get(self.last_item, '')
            if has_wake_name(transcript):
                # Addressed by name: silence is not allowed.
                self.send({'type': 'conversation.item.create', 'item': {
                    'type': 'function_call_output', 'call_id': call_id,
                    'output': 'The user addressed you by name; you must answer.'}})
                self.expect_tool_response = True
                self.send({'type': 'response.create', 'response': {'metadata': OWN, 'tool_choice': 'none', 'instructions': self.reply_language()}})
            else:
                print('Not for me; staying silent.')
                self.send({'type': 'conversation.item.create', 'item': {
                    'type': 'function_call_output', 'call_id': call_id, 'output': 'silent'}})
            return
        if name in ('move_arm', 'move_legs'):
            # Claimed synchronously, before response.done can trigger the safety net.
            with self.arm_lock:
                if self.last_item:
                    (self.arm_done_items if name == 'move_arm' else self.legs_done_items).add(self.last_item)
        threading.Thread(target=self.handle_tool_call,
                         args=(call_id, name, arguments), daemon=True).start()

    def handle_tool_call(self, call_id, name, arguments):
        try:
            args = json.loads(arguments or '{}')
            result = self.run_tool(name, args)
            print('Tool {} {} -> {}'.format(name, args, ' '.join(str(result).split())[:300]))
        except Exception as exc:
            result = 'Tool {} failed: {}. Say briefly that it did not work; do not invent a result.'.format(name, exc)
            print('Tool {} FAILED: {}'.format(name, exc))
        if self.stop.is_set():
            return
        try:
            self.send({'type': 'conversation.item.create', 'item': {
                'type': 'function_call_output', 'call_id': call_id, 'output': str(result)[:4000]}})
            if str(result).startswith(SILENT_RESULT):
                return                                  # she is shouting while moving
            self.expect_tool_response = True
            self.send({'type': 'response.create', 'response': {'metadata': OWN, 'instructions': self.reply_language() + (
                'The tool has returned. Answer the user NOW, in first person as Aisana, '
                'using its result. No commentary about tools, photos or analysis. '
                'Follow any instruction in the tool result. Otherwise one short sentence. '
                'No filler, no wishes, no offers of more help, no follow-up questions. '
                'If the tool failed, say briefly that it did not work.') }})
        except Exception as exc:
            if not self.stop.is_set():
                print('Tool response error:', exc)

    def load_history(self):
        """Earlier conversations (shared with robot.py) as context for this session."""
        if self.history is None:
            return
        try:
            turns = self.history.recent_turns(HISTORY_TURNS)
        except Exception as exc:
            print('Conversation memory unavailable:', exc)
            return
        if not turns:
            return
        now = time.time()
        lines = []
        for created, question, answer in turns:
            days = int((now - created) // 86400)
            when = 'today' if days == 0 else 'yesterday' if days == 1 else '{} days ago'.format(days)
            lines.append('[{}] User: {} | Aisana: {}'.format(when, question[:200], answer[:200]))
        text = 'Earlier conversations (use them if the user refers to the past; do not bring them up by yourself):\n'
        self.send({'type': 'conversation.item.create', 'item': {
            'type': 'message', 'role': 'system', 'content': [{'type': 'input_text', 'text': (text + '\n'.join(lines))[-3000:]}]}})
        print('Conversation memory: {} earlier turns loaded.'.format(len(turns)))

    def save_turn(self, answer):
        """Keep a real exchange (not noise, not a cut reply) for later sessions."""
        item = self.last_item
        question = self.transcripts.get(item, '')
        if (self.history is None or not question or item in self.saved_items
                or item not in self.responded_items):
            return
        self.saved_items.add(item)
        try:
            self.history.remember_turn(question, answer)
        except Exception as exc:
            print('Conversation memory error:', exc)

    def clear_history(self, item_id):
        with self.arm_lock:
            self.responded_items.add(item_id)
            self.local_items.add(item_id)
        self.saved_items.add(item_id)
        self.cancel_reply()
        if self.history is not None:
            self.history.clear_history()
        print('Conversation memory cleared.')

        def worker():
            self.speak_report('Robot report: all saved conversations were deleted.',
                              confirm_instruction('Всё, наши разговоры стёрты. Начнём с чистого листа.'
                                                  if self.user_language == 'Russian' else
                                                  'Done, our conversations are erased.', ''))
        threading.Thread(target=worker, daemon=True).start()

    def stt_fallback(self, reason):
        """The newer STT model was refused (not enabled for the key): use the old one."""
        if self.stt_model == STT_FALLBACK or not re.search(r'model|support|invalid|not found|access', reason, re.I):
            return
        print('STT model {} not available; using {}.'.format(self.stt_model, STT_FALLBACK))
        self.stt_model = STT_FALLBACK
        self.send({'type': 'session.update', 'session': {'type': 'realtime', 'audio': {'input': {
            'transcription': {'model': self.stt_model, 'prompt': STT_PROMPT}}}}})

    def instructions(self):
        prompt = PROMPT
        if FIXED_LANGUAGE:
            prompt = prompt.replace(
                'LANGUAGE: reply in the language of the user\'s latest words (Russian, Kazakh, English, Arabic). ',
                'LANGUAGE: always speak {0} only, even when the user speaks Russian, Kazakh or another '
                'language (you understand them, you answer in {0}). '.format(FIXED_LANGUAGE))
        if time.monotonic() < self.vip_until:
            return (prompt + 'GUESTS: senior university guests are here. Be polite, warm and respectful; '
                    'no sarcasm, no teasing, no jokes about anyone. ' + voice_style_instructions())
        return prompt + humor_instructions(self.humor) + voice_style_instructions()

    def guest_title(self, name):
        getter = getattr(self.faces, 'title', None)
        return getter(name) if getter else None

    def vip_seen(self, name):
        """A titled guest is in view: respectful mode for VIP_MINUTES (0 = off)."""
        if VIP_MINUTES <= 0:
            return
        starting = time.monotonic() >= self.vip_until
        self.vip_until = time.monotonic() + VIP_MINUTES * 60
        if starting:
            print('Guest in view ({}): respectful mode, no sarcasm.'.format(name))
            try:
                self.send({'type': 'session.update', 'session': {'type': 'realtime', 'instructions': self.instructions()}})
            except Exception as exc:
                print('Session update error:', exc)

    def set_humor(self, item_id, value):
        """"Айсана, юмор 60 процентов" - changes her personality live."""
        self.humor = max(0, min(100, value))
        print('Humor setting: {}%'.format(self.humor))
        with self.arm_lock:
            self.responded_items.add(item_id)
            self.local_items.add(item_id)
        self.dialog_until = time.monotonic() + DIALOG_TIMEOUT
        self.cancel_reply()
        self.send({'type': 'session.update', 'session': {'type': 'realtime', 'instructions': self.instructions()}})
        self.expect_tool_response = True
        self.send({'type': 'response.create', 'response': {'metadata': OWN, 'tool_choice': 'none', 'instructions': self.reply_language() + (
            'The user set your humor level to {}%. Confirm it in ONE short sentence in the new style; you may say the number this once.'.format(
                self.humor))}})

    def start_prefetch(self):
        """Look while the user is still speaking: "кто перед тобой?" is then
        answered about 2.5 s sooner (the vision call is already done)."""
        now = time.monotonic()
        # Only inside a conversation: in a noisy room every stranger's sentence
        # started a paid vision call (~3100 tokens) that nobody used.
        if (not CAMERA_PREFETCH or self.camera_stream is None or self.motion_active()
                or not self.dialog_active() or now - self.prefetch_at < 3):
            return
        self.prefetch_at = now
        box = {'done': threading.Event(), 'text': None}
        self.prefetch = box

        def worker():
            try:
                box['text'] = self.inspect_camera(
                    'Кто и что сейчас передо мной? Кратко: люди, их одежда и поза, что в руках, '
                    'главные предметы.', detail='low', quiet=True)   # logged only if used
            except Exception as exc:
                print('Camera prefetch failed:', exc)
            finally:
                box['done'].set()
        threading.Thread(target=worker, daemon=True).start()

    def prefetched(self, transcript):
        """The look taken while this question was spoken, if it fits the question."""
        box = self.prefetch
        if box is None or not PREFETCH_FITS.search(transcript) or READ_RE.search(transcript) \
                or time.monotonic() - self.prefetch_at > 15:
            return None
        box['done'].wait(8)
        self.prefetch = None
        return box['text']

    def run_camera_locally(self, item_id, transcript):
        """Visual question: look at once and answer in first person, no "секунду, посмотрю"."""
        with self.arm_lock:
            self.responded_items.add(item_id)
            self.local_items.add(item_id)
        self.dialog_until = time.monotonic() + DIALOG_TIMEOUT
        self.cancel_reply()

        def worker():
            try:
                seen = self.prefetched(transcript)
                if seen:
                    known = self.people_report()
                    print('Camera (looked while you spoke) -> {}{}'.format(' '.join(seen.split())[:300],
                                                                          ' | ' + known if known else ''))
                else:
                    seen = self.inspect_camera(transcript[:500])
                    print('Camera (from your words) -> {}'.format(' '.join(seen.split())[:300]))
                report = 'What you see with your own eyes right now: ' + seen + ' ' + self.people_report()
            except Exception as exc:
                print('Camera FAILED: {}'.format(exc))
                report = 'Your eyes (camera) are not working right now: {}'.format(exc)
            deadline = time.monotonic() + 8
            while self.responding and time.monotonic() < deadline and not self.stop.is_set():
                time.sleep(0.05)
            if self.stop.is_set():
                return
            if self.responding:
                self.cancel_reply()
                time.sleep(0.3)
            try:
                self.send({'type': 'conversation.item.create', 'item': {
                    'type': 'message', 'role': 'system', 'content': [{'type': 'input_text', 'text': report}]}})
                self.expect_tool_response = True
                self.send({'type': 'response.create', 'response': {'metadata': OWN, 'tool_choice': 'none', 'instructions': self.reply_language() + (
                    'Answer the user\'s question in FIRST PERSON, as if you see it with your own eyes '
                    '("Вижу...", "Передо мной..."). If face memory names someone, use that name and never '
                    'say you do not know who it is. Never mention a photo, picture, image, snapshot or '
                    'camera. In your usual character. One or two short sentences, no questions.')}})
            except Exception as exc:
                if not self.stop.is_set():
                    print('Camera report error:', exc)
        threading.Thread(target=worker, daemon=True).start()

    def speak_report(self, report, instructions):
        """Tell the model what the robot just did and let it say it in one phrase."""
        deadline = time.monotonic() + 8
        while self.responding and time.monotonic() < deadline and not self.stop.is_set():
            time.sleep(0.05)
        if self.stop.is_set():
            return
        if self.responding:
            self.cancel_reply()
            time.sleep(0.3)
        try:
            self.send({'type': 'conversation.item.create', 'item': {
                'type': 'message', 'role': 'system', 'content': [{'type': 'input_text', 'text': report}]}})
            self.expect_tool_response = True
            self.send({'type': 'response.create', 'response': {'metadata': OWN, 'tool_choice': 'none', 'instructions': (
                self.reply_language() + instructions)}})
        except Exception as exc:
            if not self.stop.is_set():
                print('Report error:', exc)

    def say_phrase(self, report, kind, name=''):
        """A memory event: the report for the model's context plus a fixed phrase."""
        phrase = memory_phrase(kind, self.user_language, name)
        instructions = confirm_instruction(phrase, '')
        if self.user_language not in MEMORY_PHRASES:
            instructions = 'Say this in {}, nothing else: "{}"'.format(self.user_language, phrase)
        self.speak_report('Robot report (true, stored permanently on the robot): ' + report, instructions)

    def face_vectors(self, seconds=3.0):
        """Vectors of the biggest face over a few seconds (several angles)."""
        vectors, frames = [], 0
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not self.stop.is_set():
            frame = self.camera_stream.latest(max_age=1.0, wait=1.0) if self.camera_stream else None
            if frame is not None:
                frames += 1
                found = self.faces.faces(frame)
                if found:
                    vectors.append(found[0]['vector'])
            time.sleep(0.25)
        print('Face views: {} of {} frames had a face'.format(len(vectors), frames))
        return vectors

    def run_memory_locally(self, item_id, command):
        """"Запомни меня, я Али", "забудь меня", "кого ты знаешь", "запомни, это моя кружка"."""
        with self.arm_lock:
            self.responded_items.add(item_id)
            self.local_items.add(item_id)
        self.dialog_until = time.monotonic() + DIALOG_TIMEOUT
        self.cancel_reply()
        transcript = self.transcripts.get(item_id, '')

        def worker():
            action, name = command['action'], command.get('name')
            ready = getattr(self.faces, 'ready', None)
            if self.faces is None or self.camera_stream is None or (ready is not None and not ready.is_set()):
                self.say_phrase('face memory is switched off or its models are still loading.', 'off')
                return
            sure = getattr(self.faces, 'threshold', 1.0) - 0.1   # renaming/deleting needs a sure match
            try:
                if action in ('remember_me', 'remember_other') and not name:
                    self.pending_face = (action, time.monotonic() + 20)
                    self.say_phrase('to remember a face the name is needed.', 'ask_name')
                    return
                if action in ('remember_me', 'remember_other'):
                    vectors = self.face_vectors()
                    if len(vectors) < 3:
                        self.say_phrase('no face was clearly visible, nothing remembered.', 'no_face')
                        return
                    if action == 'remember_me':
                        old, distance = self.faces.identify(vectors[0])
                        if old and old != name and distance < sure:
                            self.faces.forget(old)          # a person renames only themselves
                            print('Face renamed: {} -> {}'.format(old, name))
                    self.faces.remember(name, vectors)
                    now = time.monotonic()
                    self.seen_at[name] = self.greeted_at[name] = now
                    print('Face remembered: {} ({} views)'.format(name, len(vectors)))
                    self.say_phrase('the face of {} is now remembered.'.format(name), 'remembered', name)
                elif action == 'forget_me':
                    frame = self.camera_stream.latest()
                    found = self.faces.faces(frame) if frame is not None else []
                    known = found[0]['name'] if found and found[0]['distance'] < sure else None
                    if known:
                        self.faces.forget(known)
                        print('Face forgotten:', known)
                        self.say_phrase('the face of {} was deleted.'.format(known), 'forgot', known)
                    else:
                        self.say_phrase('this face was not known.', 'unknown')
                elif action == 'forget':
                    gone = self.faces.forget(name)
                    print('Forgotten: {} ({} records)'.format(name, gone))
                    if gone:
                        self.say_phrase('{} was deleted from memory.'.format(name), 'forgot_name', name)
                    else:
                        self.say_phrase('nothing called {} in memory.'.format(name), 'no_such', name)
                elif action == 'list':
                    known = self.faces.names() + ['{}{}'.format(label, ' ({})'.format(owner) if owner else '')
                                                  for _, label, owner, _ in self.faces.things]
                    if known:
                        self.say_phrase('known: ' + ', '.join(known), 'list', ', '.join(known))
                    else:
                        self.say_phrase('nobody is known yet.', 'list_none')
                elif action == 'remember_thing':
                    self.remember_thing(command.get('label') or '', transcript)
            except Exception as exc:
                print('Face memory FAILED:', exc)
                self.say_phrase('memory failed: {}'.format(exc), 'failed')
        threading.Thread(target=worker, daemon=True).start()

    def remember_thing(self, label, transcript):
        picture, frame = self.take_photo()
        owner = None
        if POSSESSIVE_RE.search(label) and frame is not None:
            found = self.faces.faces(frame)
            owner = found[0]['name'] if found and found[0]['name'] else None
        clean = ' '.join(THING_FILLER_RE.sub(' ', label).split()) or 'вещь'
        description = self.inspect_camera(
            'Describe the object the user is showing or holding (the "{}") in ONE short sentence '
            'so it can be recognized later: colour, shape, logos or text on it. Start with what it is.'.format(clean),
            extra='')
        self.faces.remember_thing(clean, owner, description,
                                  aisana_faces.jpeg(frame, 60) if frame is not None else picture)
        print('Thing remembered: {} ({}) - {}'.format(clean, owner or 'no owner', description[:120]))
        what = clean + (' ({})'.format(owner) if owner else '')
        self.say_phrase('the thing "{}" is remembered: {}'.format(what, description), 'thing', what)

    def watch_faces(self):
        """Greets people she knows when they come into view (at most once per half hour)."""
        while not self.stop.wait(FACE_SCAN_SECONDS):
            busy = (self.responding or self.playing or self.echo_guard_active or self.motion_active()
                    or self.hold_item is not None or time.monotonic() - self.speech_started_at < 3
                    or not self.ready)
            if busy:
                continue
            frame = self.camera_stream.latest(max_age=1.0, wait=0)
            if frame is None:
                continue
            try:
                found = self.faces.faces(frame)
            except Exception as exc:
                print('Face memory error:', exc)
                continue
            now = time.monotonic()
            for face in found:
                name = face['name']
                if not name:
                    continue
                self.faces.learn(name, face['vector'], face['distance'])
                title = self.guest_title(name)
                if title:
                    self.vip_seen(name)
                away = now - self.seen_at.get(name, -1e9)
                self.seen_at[name] = now
                if FACE_GREET and away > 120 and now - self.greeted_at.get(name, -1e9) > FACE_GREET_MINUTES * 60:
                    self.greeted_at[name] = now
                    print('Face seen: {} -> greeting'.format(name))
                    self.dialog_until = now + DIALOG_TIMEOUT
                    if title:
                        self.say_phrase('{} ({}) just came into view; greet them respectfully.'.format(name, title),
                                        'greet_guest', '{}, {}'.format(name, title))
                    else:
                        self.say_phrase('{} (you know them) just came into view.'.format(name), 'greet', name)
                    break

    def wave_hello(self, seconds=None):
        """Greeting gesture: right arm up, hand waves, then the arm goes back down."""
        arm = self.arm

        def worker():
            try:
                arm.execute_many([make_intent('hand_shake', 'right')], verify=False)
                print('Arms: waving hello.')
                self.end_wave_after(seconds)
            except Exception as exc:
                print('Arms: wave failed:', exc)
        threading.Thread(target=worker, daemon=True).start()

    def run_dance_locally(self, item_id, transcript):
        """Arms up and waving while she spins with shouts; afterwards arms down.
        Time from the words ("танцуй 8 секунд", at most 10) or AISANA_DANCE_SECONDS."""
        with self.arm_lock:
            self.local_items.add(item_id)
            self.responded_items.add(item_id)
            self.arm_done_items.add(item_id)
            self.legs_done_items.add(item_id)
        self.dialog_until = time.monotonic() + DIALOG_TIMEOUT
        self.cancel_reply()
        found = WAVE_FOR.search(transcript or '')
        seconds = min(aisana_legs.MAX_SECONDS, float(found.group(1)) if found else DANCE_SECONDS)

        def worker():
            done = []
            if self.arm is None:
                self.reconnect_arm()
            arm = self.arm
            if arm is not None:
                try:
                    self.wave_id += 1
                    arm.execute_many([make_intent('position', 'both', 3, 'shoulder'),
                                      make_intent('hand_shake', 'right')], verify=False)
                    done.append('arms up, right hand waving')
                    self.end_wave_after(seconds + 0.5, finish=[make_intent('hand_still', 'right'),
                                                               make_intent('position', 'both', 0)])
                except Exception as exc:
                    print('Dance: arms failed:', exc)
            spun = None
            if self.legs is not None:
                try:
                    spun = self.legs.execute(aisana_legs.make_intent('spin_right', seconds=seconds),
                                             self.report_legs_problem)
                    done.append('legs: ' + spun)
                except Exception as exc:
                    print('Dance: legs failed:', exc)
            print('Dance (from your words) -> ' + ('; '.join(done) or 'nothing connected'))
            moving = re.search(r'Driving \S+(?: \S+)? for ([\d.]+) s', spun or '')
            if moving:
                self.motion_until = time.monotonic() + float(moving.group(1)) + 1.0
                self.shouts.play_for(float(moving.group(1)))
            elif done:
                self.speak_report('Robot report: dancing with the arms only (no wheels).',
                                  confirm_instruction('Танцую!' if self.user_language == 'Russian'
                                                      else "I'm dancing!", ''))
            else:
                self.speak_report('Robot report: neither arms nor wheels are connected, no dance.',
                                  self.reply_language() + 'In ONE short sentence say that you cannot dance now.')
        threading.Thread(target=worker, daemon=True).start()

    def end_wave_after(self, seconds=None, finish=None):
        """Stop waving and lower the right arm after a while, unless another arm
        command (or a stop) came first. Nov 23: the wave had no timer and the
        greeting ended too early; default 6 s, a spoken time works too."""
        self.wave_id += 1
        mine = self.wave_id
        seconds = WAVE_SECONDS if seconds is None else seconds
        arm = self.arm

        def worker():
            if self.stop.wait(seconds) or mine != self.wave_id or arm is None:
                return
            try:
                arm.execute_many(finish or [make_intent('hand_still', 'right'),
                                            make_intent('position', 'right', 0)], verify=False)   # arm down again
                print('Arms: waving done, arms down.')
            except Exception as exc:
                print('Arms: could not end the wave:', exc)
        threading.Thread(target=worker, daemon=True).start()

    def reconnect_arm(self):
        """The Arduino was missing or busy at start (Jetson log, Nov 20: "the arm
        controller is not connected"): try again when an arm command comes,
        at most every 15 s. Opening the port resets the board (~3 s)."""
        now = time.monotonic()
        if now - self.arm_retry_at < 15:
            return
        self.arm_retry_at = now
        print('Arm: not connected; trying again...')
        self.arm = connect_from_environment()
        if self.arm is not None and self.on_arm_connected is not None:
            self.on_arm_connected(self.arm)

    def motion_active(self):
        return time.monotonic() < self.motion_until

    def stop_motion(self):
        self.motion_until = 0.0
        self.shouts.stop()

    def holding(self):
        if self.hold_item is None:
            return False
        if time.monotonic() > self.hold_deadline:
            self.release_hold(drop=False)            # STT is late: do not stay silent forever
            return False
        return self.response_item == self.hold_item

    def release_hold(self, drop):
        held, self.held_audio, self.hold_item = self.held_audio, [], None
        if not drop:
            for kind, chunk in held:
                self.queue_audio(kind, chunk)

    def reply_language(self):
        if FIXED_LANGUAGE:
            return 'Reply in {} only, even if the user speaks another language. '.format(FIXED_LANGUAGE)
        return 'Reply in {} only (the language of the user\'s latest words). '.format(self.user_language)

    def handle_transcript(self, item_id, transcript):
        print('You:', transcript)
        if not FIXED_LANGUAGE:
            self.user_language = language_of(transcript) or self.user_language
        with self.transcript_cond:
            self.transcripts[item_id] = transcript
            self.transcript_cond.notify_all()
        held = self.hold_item == item_id
        if not re.search(r'\w', transcript or '') or is_prompt_echo(transcript) \
                or SHOUT_ECHO_RE.match(transcript.strip()):
            # Noise or the STT hint repeated back: never a request, never a move.
            print('Empty speech or noise; ignored.')
            if held:
                self.release_hold(drop=True)
            if item_id in self.responded_items:
                self.cancel_reply()
            self.ignore_item(item_id)
            return
        if WAKE_GATE and not has_wake_name(transcript) and is_stray_word(transcript):
            # "Pairing." heard in an open dialogue: noise or someone else, not a reply.
            print('Stray word; ignored.')
            if held:
                self.release_hold(drop=True)
            if item_id in self.responded_items:
                self.cancel_reply()
            self.ignore_item(item_id)
            return
        full = transcript
        transcript = request_part(transcript)
        trimmed = transcript != full
        if trimmed:
            print('Before her name (ignored): ' + full[:len(full) - len(transcript)].strip())
        arm_intents = parse_arm_commands(transcript)
        drive = None if arm_intents else parse_legs_command(transcript)
        if not arm_intents and drive is None and REPEAT_RE.search(transcript) and self.last_drive \
                and time.monotonic() - self.last_drive_at < 180:
            drive = dict(self.last_drive)                # "повтори", "ещё раз", "повтори 10 раз"
            times = TIMES_RE.search(transcript.casefold())
            if times:
                count = TIMES_WORDS.get(times.group(1)) or int(times.group(1))
                if drive['action'] in ('spin_left', 'spin_right'):
                    drive['degrees'] = min(3600, (drive.get('degrees') or 90) * count)
                    drive['seconds'] = None
                elif drive.get('seconds'):
                    drive['seconds'] = drive['seconds'] * count
        if self.motion_active() and not (has_wake_name(transcript) and (arm_intents or drive)):
            # While the wheels turn she hears her own shouts: only "стоп" counts.
            if is_stop_phrase(transcript) or (drive and drive['action'] == 'stop'):
                self.stop_motion()
                if self.legs is not None:
                    self.legs.stop()
                if self.arm is not None:
                    self.wave_id += 1           # no arm move after a stop
                    self.arm.emergency_stop()
                print('Stopped during motion.')
            if held:
                self.release_hold(drop=True)
            self.ignore_item(item_id)
            return
        stop = (is_stop_phrase(transcript) or (drive and drive['action'] == 'stop')
                or (arm_intents and arm_intents[0]['action'] == 'stop'))
        if stop:
            self.stop_motion()
            if self.legs is not None:
                self.legs.stop()            # wheels first: they can run into people
            if self.arm is not None:
                self.wave_id += 1           # no arm move after a stop
                self.arm.emergency_stop()
            self.stop_playback()
        addressed = (not WAKE_GATE or item_id in self.responded_items
                     or self.dialog_active() or has_wake_name(transcript))
        if stop and addressed:
            # Already stopped above. Jetson log, Nov 23: the model then called
            # move_arm stop again (a second "x") and said "Опускаю руки в ноль".
            if held:
                self.release_hold(drop=True)
            with self.arm_lock:
                self.local_items.add(item_id)
                self.responded_items.add(item_id)
            self.cancel_reply()
            phrase = {'Russian': 'Стою.', 'English': 'Stopped.', 'Kazakh': 'Тоқтадым.',
                      'Egyptian Arabic': 'وقفت.'}.get(self.user_language, 'Stopped.')
            threading.Thread(target=self.speak_report, daemon=True, args=(
                'Robot report: wheels and arms were stopped.', confirm_instruction(phrase, ''))).start()
            return
        if addressed and not stop and DANCE_RE.search(transcript) and not re.search(r'\?\s*$', transcript):
            if held:
                self.release_hold(drop=True)
            self.run_dance_locally(item_id, transcript)
            return
        if addressed and not stop and (arm_intents or drive):
            # A clear command: run it now from the user's exact words and only
            # confirm it, instead of letting the model think aloud first.
            if held:
                self.release_hold(drop=True)
            self.run_command_locally(item_id, arm_intents, drive)
            return
        if addressed and not stop and self.arm is not None and GREET_RE.search(transcript):
            self.wave_hello()                       # the reply itself comes from the model below
        humor = HUMOR_RE.search(transcript) if addressed and not stop else None
        if humor:
            if held:
                self.release_hold(drop=True)
            self.set_humor(item_id, int(humor.group(1)))
            return
        if addressed and not stop and CLEAR_HISTORY_RE.search(transcript):
            if held:
                self.release_hold(drop=True)
            self.clear_history(item_id)
            return
        memory = parse_memory_command(transcript) if addressed and not stop else None
        if memory is None and self.pending_face and time.monotonic() < self.pending_face[1] \
                and self.dialog_active() and not stop:
            # "Как тебя зовут?" -> "Али" / "меня зовут Али"
            found = aisana_faces.SELF_NAME_RE.search(transcript)
            name = aisana_faces._name(found.group(1) if found else transcript)
            if name and len(transcript.split()) <= 4:
                memory = {'action': self.pending_face[0], 'name': name, 'label': None}
        if memory:
            self.pending_face = None
            if held:
                self.release_hold(drop=True)
            self.run_memory_locally(item_id, memory)
            return
        if addressed and not stop and CAMERA_RE.search(transcript):
            if held:
                self.release_hold(drop=True)
            self.run_camera_locally(item_id, transcript)
            return
        if trimmed and addressed and not stop:
            if held:
                self.release_hold(drop=True)
            self.answer_request_only(item_id, transcript)
            return
        if held:
            self.release_hold(drop=False)              # an ordinary question: play the answer
        if not WAKE_GATE or item_id in self.responded_items:
            return
        if is_sleep_phrase(transcript):
            self.dialog_until = 0.0
            print('Dialogue closed; say Aisana to begin again.')
            self.ignore_item(item_id)
        elif addressed:
            self.respond(item_id)
        else:
            print('Not addressed to Aisana; ignored.')
            self.ignore_item(item_id)

    def cancel_reply(self):
        if self.responding:
            self.cut_ids.add(self.response_id)
            try:
                self.send({'type': 'response.cancel'})
            except Exception:
                pass
        self.stop_playback()

    def run_command_locally(self, item_id, arm_intents, drive):
        with self.arm_lock:
            if item_id in self.arm_done_items or item_id in self.legs_done_items:
                return                      # the model already called the tool for it
            self.arm_done_items.add(item_id)
            self.legs_done_items.add(item_id)
            self.responded_items.add(item_id)
            self.local_items.add(item_id)
        self.dialog_until = time.monotonic() + DIALOG_TIMEOUT
        self.cancel_reply()                 # cut "Секунду, подумаю..." chatter

        def worker():
            try:
                if arm_intents:
                    if self.arm is None:
                        self.reconnect_arm()
                    if self.arm is None:
                        result = 'NOT done: the arm controller is not connected over USB.'
                    else:
                        self.wave_id += 1                   # any arm command cancels a wave timer
                        self.arm.execute_many(arm_intents, verify=False)
                        self.arm.watch(arm_intents, self.report_arm_problem)
                        if arm_intents[-1]['action'] == 'hand_shake':
                            self.end_wave_after(wave_seconds(self.transcripts.get(item_id, '')))
                        self.last_arm_at = time.monotonic()
                        result = 'Arms: ' + describe(arm_intents) + '.'
                elif self.legs is None:
                    result = 'NOT done: legs are not connected (AISANA_LEGS_URL).'
                else:
                    result = 'Legs: ' + self.legs.execute(drive, self.report_legs_problem)
                    if drive['action'] not in ('stop', 'faster', 'slower'):
                        self.last_drive, self.last_drive_at = dict(drive), time.monotonic()
            except Exception as exc:
                result = 'NOT done: {}'.format(exc)
            print('Command (from your words) -> {}'.format(' '.join(result.split())[:300]))
            deadline = time.monotonic() + 3
            while self.responding and time.monotonic() < deadline and not self.stop.is_set():
                time.sleep(0.05)            # let the cancelled reply finish
            if self.stop.is_set():
                return
            moving = re.search(r'Driving \S+(?: \S+)? for ([\d.]+) s', result)
            transcript = self.transcripts.get(item_id, '')
            if moving and not shout_request(transcript):
                seconds = float(moving.group(1))
                self.motion_until = time.monotonic() + seconds + 1.0
                if self.shouts.play_for(seconds):
                    # Only her shouts while the wheels turn: no words at all.
                    self.send({'type': 'conversation.item.create', 'item': {
                        'type': 'message', 'role': 'system', 'content': [{'type': 'input_text', 'text': (
                            'Robot report (no reply needed): ' + result)}]}})
                    return
            try:
                self.send({'type': 'conversation.item.create', 'item': {
                    'type': 'message', 'role': 'system', 'content': [{'type': 'input_text', 'text': (
                        'Robot report: the user\'s command was executed by the robot itself. ' + result)}]}})
                self.expect_tool_response = True
                if result.startswith('NOT done'):
                    instructions = self.reply_language() + (
                        'In ONE short sentence say what is not working, from the robot report. Nothing else.')
                else:
                    # A fixed short phrase: commands get "Еду вперёд.", turnarounds "Уииииии!".
                    phrase = (arm_reply(arm_intents[-1], self.user_language) if arm_intents
                              else legs_reply(drive, self.user_language))
                    instructions = confirm_instruction(phrase, self.transcripts.get(item_id, ''))
                self.send({'type': 'response.create', 'response': {'metadata': OWN, 'tool_choice': 'none', 'instructions': instructions}})
            except Exception as exc:
                if not self.stop.is_set():
                    print('Command report error:', exc)
        threading.Thread(target=worker, daemon=True).start()

    def handle(self, e):
        typ = e.get('type', '')
        if typ == 'session.updated':
            self.ready = True
            print('Realtime session ready. Talk to Aisana (Ctrl+C to stop).')
            if self.shouts.pcm is None and not self.oob_ids:
                # Record her own shouts once, silently (out-of-band, not played).
                self.send({'type': 'response.create', 'response': {
                    'conversation': 'none', 'output_modalities': ['audio'],
                    'metadata': {'purpose': 'shouts'}, 'instructions': SHOUTS_PROMPT}})
        elif typ == 'input_audio_buffer.speech_started':
            self.speech_started_at = time.monotonic()
            if self.spotter is not None and time.monotonic() < self.stream_until:
                self.stream_until = time.monotonic() + 20     # a long question keeps streaming
            self.start_prefetch()
            if BARGE_IN and (self.responding or self.playing):
                self.playback_interrupted = True
                self.stop_playback()
                # Realtime server_vad with interrupt_response=True already
                # cancels the active generation. A second response.cancel races
                # with response.done and triggers response_cancel_not_active.
        elif typ == 'response.created' and ((e.get('response') or {}).get('metadata') or {}).get('purpose') == 'shouts':
            self.oob_ids.add((e.get('response') or {}).get('id'))
            self.shout_buffer = bytearray()
        elif typ == 'response.created':
            purpose = ((e.get('response') or {}).get('metadata') or {}).get('purpose')
            self.responding = True
            self.response_id = (e.get('response') or {}).get('id')
            if purpose != 'own' and self.last_item in self.local_items:
                # A model reply to words the code already handled (the camera, a
                # command): it arrived after we tried to cancel. Cut it, never play it.
                # Jetson log: only "Сейчас быстро проверю..." and the real answer lost.
                self.response_item = None
                self.playback_interrupted = True
                self.cut_ids.add(self.response_id)
                try:
                    self.send({'type': 'response.cancel'})
                except Exception:
                    pass
                return
            self.shouts.stop()                 # the speaker is needed for a real reply
            if purpose == 'own' or self.expect_tool_response:
                self.expect_tool_response = False
                self.response_item = None      # answer to a tool result
            else:
                self.response_item = self.last_item
            self.echo_guard_active = True
            self.playback_interrupted = False
        elif typ == 'response.output_audio.delta' and e.get('response_id') in self.oob_ids:
            self.shout_buffer.extend(base64.b64decode(e.get('delta', '')))     # recorded, not played
        elif typ == 'response.output_audio.delta':
            self.echo_guard_active = True
            if not self.playback_interrupted:
                chunk = base64.b64decode(e.get('delta', ''))
                if self.holding():
                    self.held_audio.append(('audio', chunk))
                else:
                    self.queue_audio('audio', chunk)
        elif typ == 'response.output_audio_transcript.done' and e.get('response_id') in self.oob_ids:
            print('Shouts recorded:', ' '.join((e.get('transcript') or '').split()))
        elif typ == 'response.output_audio_transcript.done':
            # One clean console line per reply (audio itself still streams).
            text = ' '.join((e.get('transcript') or '').split())
            if text:
                cut = e.get('response_id') in self.cut_ids
                print('Aisana (cut):' if cut else 'Aisana:', text)
                if not cut:
                    self.save_turn(text)
        elif typ == 'input_audio_buffer.committed':
            self.last_item = e.get('item_id')
            # Inside an open dialogue start the answer at once, but hold its audio
            # until the transcript shows it is not a command (or 1.6 s passed).
            # While the wheels turn, wait for the transcript (her shouts are heard).
            if WAKE_GATE and self.dialog_active() and not self.motion_active():
                self.hold_item, self.held_audio = e.get('item_id'), []
                self.hold_deadline = time.monotonic() + 1.6
                self.respond(e.get('item_id'))
        elif typ == 'conversation.item.input_audio_transcription.completed':
            self.handle_transcript(e.get('item_id'), (e.get('transcript') or '').strip())
        elif typ == 'conversation.item.input_audio_transcription.failed':
            print('Realtime transcription failed:', e.get('error', {}))
            self.stt_fallback(str(e.get('error', {})))
        elif typ == 'response.function_call_arguments.done':
            self.start_tool(e.get('call_id'), e.get('name'), e.get('arguments'))
        elif typ == 'response.output_item.done':
            item = e.get('item') or {}
            if item.get('type') == 'function_call':
                self.start_tool(item.get('call_id'), item.get('name'), item.get('arguments'))
        elif typ == 'response.done' and (e.get('response') or {}).get('id') in self.oob_ids:
            self.oob_ids.discard((e.get('response') or {}).get('id'))
            if len(self.shout_buffer) >= RATE:                 # at least 0.5 s of shouting
                self.shouts.pcm = bytes(self.shout_buffer)
                print('Shouts ready: {:.1f} s'.format(len(self.shout_buffer) / (RATE * 2.0)))
        elif typ == 'response.done':
            self.responding = False
            holding = self.holding()
            if self.response_item:
                self.done_items.add(self.response_item)
                self.run_pending_arm(self.response_item)
                self.run_pending_legs(self.response_item)
            # Do NOT lift echo guard here: playback can lag behind generation.
            if holding:
                self.held_audio.append(('done', b''))
            else:
                self.queue_audio('done')
        elif typ == 'error':
            error = e.get('error', {}) or {}
            if error.get('code') == 'response_cancel_not_active':
                return                          # harmless: the reply had already ended
            print('Realtime API error:', error)
            if re.search(r'transcri', str(error), re.I):
                self.stt_fallback(str(error))
            if self.oob_ids and re.search(r'conversation|metadata|out.of.band', str(error.get('message', '')), re.I):
                self.oob_ids.clear()                    # shout recording unsupported: fall back
                print('Shout recording not available; she will shout once per move instead.')

    def run(self, manage_devices=True):
        """One realtime session. With manage_devices=False the caller owns the
        arm/legs links (kept across reconnects: reopening the Arduino port
        would reset the arm board)."""
        if not KEY:
            raise RuntimeError('OPENAI_API_KEY is missing')
        if websocket is None:
            raise RuntimeError('Realtime mode needs websocket-client: python3 -m pip install --user "websocket-client==1.3.1"')
        self.mic_device = alsa_device('arecord', MIC_LABEL, 'microphone ' + MIC_LABEL)
        self.speaker = alsa_device('aplay', 'USB Composite Device', 'Xiaomi speaker')
        self.shouts.speaker = self.speaker
        print('Input:', self.mic_device, 'Output:', self.speaker)
        print('STT:', self.stt_model, 'Mic gain:', MIC_GAIN, 'Noise reduction:', NOISE_REDUCTION,
              'VAD threshold:', VAD_THRESHOLD)
        print('Model:', MODEL, 'Voice:', VOICE,
              'Barge-in:', 'ON (experimental)' if BARGE_IN else 'OFF',
              'Echo guard:', 'ON (no interruptions during playback)' if ECHO_GUARD else 'OFF',
              'Wake name:', 'required to start' if WAKE_GATE else 'OFF (answers any speech)')
        if manage_devices:
            self.arm = connect_from_environment()
            self.legs = aisana_legs.connect_from_environment()
        self.ws = websocket.create_connection(
            'wss://api.openai.com/v1/realtime?model=' + MODEL,
            timeout=10, header=['Authorization: Bearer ' + KEY])
        try:
            self.send({
                'type': 'session.update',
                'session': {
                    'type': 'realtime',
                    'model': MODEL,
                    'instructions': self.instructions(),
                    'tools': tool_definitions(),
                    'tool_choice': 'auto',
                    'output_modalities': ['audio'],
                    'audio': {
                        'input': {
                            'format': {'type': 'audio/pcm', 'rate': RATE},
                            # Transcripts drive the wake-name gate and voice "stop".
                            'transcription': {
                                'model': self.stt_model,
                                # A list of names here gets echoed back on noise.
                                # Keep it short: STT repeats its hint on noise (a command list here
                                # was once "heard" and opened the grippers).
                                'prompt': STT_PROMPT
                            },
                            'noise_reduction': {'type': NOISE_REDUCTION} if NOISE_REDUCTION in ('near_field', 'far_field') else None,
                            'turn_detection': {
                                'type': 'server_vad',
                                'threshold': VAD_THRESHOLD,
                                'prefix_padding_ms': 500,
                                'silence_duration_ms': max(500, min(3000, SILENCE_MS)),
                                'create_response': not WAKE_GATE,
                                'interrupt_response': BARGE_IN
                            }
                        },
                        'output': {
                            'format': {'type': 'audio/pcm', 'rate': RATE},
                            'voice': VOICE
                        }
                    }
                }
            })
            self.load_history()
            self.ws.settimeout(10)
            threading.Thread(target=self.audio_worker, daemon=True).start()
            if self.faces is not None and self.camera_stream is not None:
                threading.Thread(target=self.watch_faces, daemon=True).start()
            worker = threading.Thread(target=self.capture, daemon=True)
            worker.start()
            while not self.stop.is_set():
                try:
                    event = json.loads(self.ws.recv())
                except websocket.WebSocketTimeoutException:
                    continue
                self.handle(event)
        finally:
            self.stop.set()
            if self.mic:
                self.mic.terminate()
                try:
                    self.mic.wait(timeout=2)
                except Exception:
                    self.mic.kill()
                    self.mic.wait()
            self.stop_playback()
            self.shouts.stop()
            if self.legs is not None:
                self.legs.stop()            # never keep driving without a session
            if manage_devices and self.arm is not None:
                self.arm.close()
            if self.ws:
                try:
                    self.ws.close()
                except Exception:
                    pass


reconnect_pause = time.sleep        # replaced in tests


def main():
    """Run sessions forever: a dropped Wi-Fi or OpenAI connection reconnects
    by itself instead of ending the program. Ctrl+C stops everything."""
    arm = connect_from_environment()
    legs = aisana_legs.connect_from_environment()
    camera, faces = aisana_faces.connect_from_environment()     # camera stays on across reconnects
    spotter = wake_listener.spotter_from_environment()
    delay = 2
    try:
        while True:
            started = time.monotonic()
            session = RealtimeAisana()
            session.arm, session.legs = arm, legs
            holder = {}
            session.on_arm_connected = lambda link: holder.setdefault('arm', link)
            session.camera_stream, session.faces = camera, faces
            session.spotter = spotter
            try:
                session.run(manage_devices=False)
                problem = 'session ended'
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                problem = str(exc) or exc.__class__.__name__
            arm = holder.get('arm', arm)
            if not KEY or websocket is None:
                raise RuntimeError(problem)
            delay = 2 if time.monotonic() - started > 60 else min(30, delay * 2)
            print('Connection lost ({}); reconnecting in {} s... (Ctrl+C to stop)'.format(problem, delay))
            reconnect_pause(delay)
    finally:
        if legs is not None:
            legs.stop()
        if arm is not None:
            arm.close()
        if camera is not None:
            camera.stop()


def _stop_now(signum, frame):
    """Second Ctrl+C (or a stuck cleanup): leave at once. The Pi stops the
    wheels by itself 0.35 s after the last heartbeat."""
    print('\nAisana stopped (forced).')
    os._exit(1)


def _interrupt(signum, frame):
    signal.signal(signal.SIGINT, _stop_now)        # Ctrl+C again = leave at once
    signal.signal(signal.SIGTERM, _stop_now)
    raise KeyboardInterrupt


if __name__ == '__main__':
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)       # "pkill -f aisana_realtime.py"
    code = 0
    try:
        main()
    except KeyboardInterrupt:
        print('\nAisana stopped.')
    except Exception as exc:
        print('Aisana realtime error:', exc)
        code = 1
    # Do not wait for camera/audio threads: GStreamer could block a normal exit
    # (Jetson, Nov 20: Ctrl+C did not stop her).
    sys.stdout.flush()
    os._exit(code)
