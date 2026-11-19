"""Voice control of the legs (wheel drive) on the Raspberry Pi.

The Pi runs the legs team's Flask app (app.py + odrive_controller.py, port
5000). Nothing on the Pi changes: Aisana uses the same HTTP API as the
browser page /control:

    POST /motor/enable    {"calibrate": false}   connect + closed loop
    POST /motor/command   {"command": "forward"} forward, backward, arc_left,
                                                 arc_right, spin_left,
                                                 spin_right, stop
    POST /motor/heartbeat                        must arrive every <0.35 s or
                                                 the Pi stops (0.75 s: motors off)
    POST /motor/speed     {"delta": 0.5}
    GET  /motor/status

Safety built on the Pi's own interlocks:
- every drive has a time limit (default 2 s, max 10 s); Aisana sends the
  heartbeat only while driving, so the Pi stops by itself if the Jetson,
  Wi-Fi or this program dies;
- the Pi's obstacle stop (< 40 cm) answers 409 and the drive ends;
- motors are NEVER calibrated by voice (needs the robot lifted).

Python 3.6 compatible. Environment:
    AISANA_LEGS_URL=http://192.168.1.50:5000   the Pi (unset = legs disabled)
    AISANA_LEGS_SPIN_90_SECONDS=1.5            spin time for a 90 degree turn
"""
import os
import re
import threading
import time

DRIVE_COMMANDS = ('forward', 'backward', 'arc_left', 'arc_right', 'spin_left', 'spin_right')
LEGS_ACTIONS = DRIVE_COMMANDS + ('stop', 'faster', 'slower')
DEFAULT_SECONDS = 2.0
SHORT_SECONDS = 1.0
MAX_SECONDS = 10.0
HEARTBEAT_SECONDS = 0.12          # the Pi stops after 0.35 s without one
ENABLE_TIMEOUT = 12.0             # odrive.find_any() can take a while the first time


class LegsError(RuntimeError):
    """The Pi is unreachable, motors cannot be enabled, or a command is refused."""


def spin_90_seconds(environ=None):
    environ = os.environ if environ is None else environ
    try:
        return max(0.2, min(5.0, float(environ.get('AISANA_LEGS_SPIN_90_SECONDS', '1.5'))))
    except ValueError:
        return 1.5


def _intent(action, seconds=None, meters=None, degrees=None):
    return {'action': action, 'seconds': seconds, 'meters': meters, 'degrees': degrees}


def plan_seconds(intent, speed_kmh=2.0, spin_90=1.5):
    """How long to drive for an intent; always within (0, MAX_SECONDS]."""
    if intent.get('seconds'):
        seconds = float(intent['seconds'])
    elif intent.get('meters'):
        seconds = float(intent['meters']) / max(0.1, float(speed_kmh) / 3.6)
    elif intent['action'] in ('spin_left', 'spin_right'):
        seconds = spin_90 * float(intent.get('degrees') or 90) / 90.0
    else:
        seconds = DEFAULT_SECONDS
    return max(0.3, min(MAX_SECONDS, seconds))


class LegsController(object):
    """HTTP client for the Pi plus one background drive at a time."""

    def __init__(self, base_url, http=None, log=print, spin_90=1.5):
        if http is None:
            import requests
            http = requests
        self.base_url = base_url.rstrip('/')
        self.http = http
        self.log = log
        self.spin_90 = spin_90
        self._cancel = threading.Event()
        self._drive = None
        self._lock = threading.Lock()

    # --- raw API -----------------------------------------------------------
    def _post(self, path, payload=None, timeout=1.0):
        response = self.http.post(self.base_url + path, json=payload or {}, timeout=timeout)
        try:
            data = response.json()
        except ValueError:
            data = {}
        return response.status_code, data

    def status(self):
        response = self.http.get(self.base_url + '/motor/status', timeout=1.0)
        return response.json()

    def ensure_enabled(self):
        """Enable motors WITHOUT calibration; raise with a clear reason if impossible."""
        status = self.status()
        if status.get('enabled'):
            return status
        self._post('/motor/enable', {'calibrate': False}, timeout=2.0)
        deadline = time.monotonic() + ENABLE_TIMEOUT
        while time.monotonic() < deadline:
            status = self.status()
            if status.get('enabled'):
                return status
            state = str(status.get('state', ''))
            if 'NEEDS CALIBRATION' in state:
                raise LegsError('motors need calibration: lift the robot and press '
                                'CALIBRATE & ENABLE on the Pi /control page')
            if state == 'ERROR':
                raise LegsError('motors could not be enabled: {}'.format(status.get('error')))
            time.sleep(0.2)
        raise LegsError('motors did not enable in time (is the ODrive powered?)')

    # --- drives ------------------------------------------------------------
    def stop(self):
        """Stop now. Safe to call any time; never raises."""
        self._cancel.set()
        try:
            self._post('/motor/command', {'command': 'stop'}, timeout=0.5)
        except Exception as error:
            # No heartbeat any more: the Pi stops on its own within 0.35 s.
            self.log('Legs: stop request failed ({}); the Pi stops by itself.'.format(error))

    def change_speed(self, delta):
        code, data = self._post('/motor/speed', {'delta': delta})
        if code != 200:
            raise LegsError(data.get('error') or 'speed change refused')
        return data.get('speed_kmh')

    def drive(self, intent, report=None):
        """Start one time-limited drive in the background; returns planned seconds.

        report(problem_text) is called only if something goes wrong
        (unreachable Pi, calibration needed, obstacle, refused command).
        """
        action = intent['action']
        if action not in DRIVE_COMMANDS:
            raise LegsError('Unknown drive: {!r}'.format(action))
        report = report or (lambda text: None)
        with self._lock:
            self._cancel.set()                       # end any previous drive
            if self._drive is not None:
                self._drive.join(1.0)
            cancel = threading.Event()
            self._cancel = cancel
        try:
            speed = float(self.status().get('speed_kmh') or 2.0)
        except Exception:
            speed = 2.0
        seconds = plan_seconds(intent, speed, self.spin_90)

        # Enable and send the first command BEFORE returning, so a refusal
        # (obstacle closer than 40 cm, motors off) is known before Aisana speaks.
        self.ensure_enabled()
        code, data = self._post('/motor/command', {'command': action})
        if code != 200:
            raise LegsError(data.get('error') or 'command refused ({})'.format(code))
        self.log('Legs: {} for {:.1f} s'.format(action, seconds))

        def worker():
            try:
                end = time.monotonic() + seconds
                while time.monotonic() < end and not cancel.is_set():
                    code, data = self._post('/motor/heartbeat', timeout=0.3)
                    if code != 200 or data.get('command') == 'stop':
                        # The Pi stopped on its own (obstacle closer than 40 cm).
                        raise LegsError('stopped by the robot: obstacle ahead or link problem')
                    cancel.wait(HEARTBEAT_SECONDS)
            except Exception as error:
                if not cancel.is_set():
                    self.log('Legs problem: {}'.format(error))
                    report(str(error))
            finally:
                try:
                    self._post('/motor/command', {'command': 'stop'}, timeout=0.5)
                except Exception:
                    pass                             # heartbeats stopped: Pi stops anyway

        thread = threading.Thread(target=worker, daemon=True)
        with self._lock:
            self._drive = thread
        thread.start()
        return seconds

    def execute(self, intent, report=None):
        """Run a parsed intent; returns a short English summary for the model."""
        action = intent['action']
        if action == 'stop':
            self.stop()
            return 'Legs stopped.'
        if action in ('faster', 'slower'):
            speed = self.change_speed(0.5 if action == 'faster' else -0.5)
            return 'Drive speed is now {} km/h.'.format(speed)
        try:
            # Enabling and the first command happen before the reply, so Aisana
            # never says "Еду" when the Pi refused (obstacle, calibration...).
            seconds = self.drive(intent, report)
        except Exception as error:
            return ('NOT driving: {}. Nothing moved. Tell the user this in one short sentence.'.format(error))
        return 'Driving {} for {:.1f} s (stops by itself).'.format(action.replace('_', ' '), seconds)


def connect_from_environment(log=print, environ=None, http=None):
    """Legs are enabled only when AISANA_LEGS_URL points at the Pi."""
    environ = os.environ if environ is None else environ
    url = environ.get('AISANA_LEGS_URL', '').strip()
    if not url:
        log('Legs: not configured (set AISANA_LEGS_URL=http://<pi-ip>:5000).')
        return None
    legs = LegsController(url, http=http, log=log, spin_90=spin_90_seconds(environ))
    try:
        status = legs.status()
        log('Legs: Pi at {} answers, motors {}.'.format(url, status.get('state', '?')))
    except Exception as error:
        log('Legs: Pi at {} not reachable ({}); will retry on each command.'.format(url, error))
    return legs


# ---------------------------------------------------------------------------
# Local parsing of spoken drive commands. A command needs a movement word;
# arm commands are handled first by aisana_arm, so "руку вперёд" never drives.
FORWARD = re.compile(r'(?:вп[её]р[её]д\b|\bforward\b|\bahead\b|алға|لقدام|قدام)', re.I)
BACKWARD = re.compile(r'(?:назад|\bback(?:wards?)?\b|артқа|لورا|ورا)', re.I)
LEFT = re.compile(r'(?:налево|влево|лево|левее|\bleft\b|солға|شمال|يسار)', re.I)
RIGHT = re.compile(r'(?:направо|вправо|право|правее|\bright\b|оңға|يمين)', re.I)
# "поведи/повели" is how STT often hears "поверни" (Jetson log, Nov 19).
SPIN = re.compile(r'(?:поверн\w*|повед[иь]\w*|повели\b|разверн\w*|крут\w*|разворот\w*|\bturn\b|\bspin\b|'
                  r'бұрыл\w*|لف)', re.I)
TIMES_RE = re.compile(r'(\d+)\s*(?:раз\w*|times|рет|مرات|مرة)', re.I)
FEW_TIMES = re.compile(r'(?:несколько раз|пару раз|много раз|a few times|several times|бірнеше рет)', re.I)
AROUND = re.compile(r'(?:разверн\w*|разворот\w*|turn around|кругом)', re.I)
DRIVE = re.compile(
    r'(?:\bедь\b|езжай|поезжай|прое[дзхй]\w*|подъе[дзхй]\w*|отъе[дзхй]\w*|ехай|едем|поехали|двигайся|'
    r'подвинься|иди\b|пойди|шагай|\bdrive\b|\bgo\b|\bmove\b|\bcome\b|\broll\b|жүр|бар\b|'
    r'وريني|امشي|روح|تعالى)', re.I)
LEG_STOP = re.compile(r'(?:\bстоп\b|останов\w*|стой|замри|\bstop\b|\bhalt\b|тоқта\w*|وقف|استنى)', re.I)
FASTER = re.compile(r'(?:быстрее|\bfaster\b|жылдам\w*|أسرع|بسرعة)', re.I)
SLOWER = re.compile(r'(?:медленнее|потише|\bslower\b|баяу\w*|بالراحة|أبطأ)', re.I)
SECONDS_RE = re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:секунд\w*|сек\b|seconds?|secs?|s\b|секунд|ثاني\w*)', re.I)
METERS_RE = re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:метр\w*|m\b|meters?|metres?|متر)', re.I)
CM_RE = re.compile(r'(\d+)\s*(?:сантиметр\w*|см\b|cm\b|centimet\w*)', re.I)
DEGREES_RE = re.compile(r'(\d{1,3})\s*(?:градус\w*|°|degrees?|درج\w*)', re.I)
A_BIT = re.compile(r'(?:немного|чуть|слегка|a bit|a little|slightly|аздап|сәл|شوية)', re.I)
METER_WORD = re.compile(r'(?:на метр|one meter|a meter|бір метр)', re.I)
LEG_CONTEXT = re.compile(r'(?:\bноги\b|колёс\w*|колес\w*|\blegs?\b|wheels?|робот\w*)', re.I)


make_intent = _intent


def mentions_driving(text):
    """Does the user talk about driving at all (for vague requests the model interprets)?"""
    q = (text or '').casefold()
    return bool(DRIVE.search(q) or LEG_CONTEXT.search(q) or FORWARD.search(q) or BACKWARD.search(q)
                or SPIN.search(q) or FASTER.search(q) or SLOWER.search(q))


def mentions_amount(text):
    q = _digits((text or '').casefold())
    return bool(SECONDS_RE.search(q) or METERS_RE.search(q) or CM_RE.search(q) or DEGREES_RE.search(q)
                or METER_WORD.search(q) or A_BIT.search(q))


NUMBER_WORDS = (
    (r'полметра', '0.5 метра'), (r'полсекунды', '0.5 секунды'), (r'пол ?минуты', '10 секунд'),
    (r'один|одну|одна|one', '1'), (r'два|две|two', '2'), (r'три|three', '3'), (r'четыре|four', '4'),
    (r'пять|five', '5'), (r'шесть|six', '6'), (r'семь|seven', '7'), (r'восемь|eight', '8'),
    (r'девять|nine', '9'), (r'десять|ten', '10'),
)


def _digits(text):
    """'пять метров' -> '5 метров' so the same regexes read both."""
    for pattern, value in NUMBER_WORDS:
        text = re.sub(r'(?<!\w)(?:' + pattern + r')(?!\w)', value, text)
    return text


def _number(text):
    return float(text.replace(',', '.'))


def parse_legs_command(text):
    """{'action', 'seconds', 'meters', 'degrees'} or None if not a drive command."""
    q = _digits(' '.join((text or '').casefold().split()))
    # The name and fillers do not count as words: "Айсана, вперёд!" is short.
    core = re.sub(r'(?:айсан\w*|эйсан\w*|aisan\w*|aysan\w*|давай|ну|пожалуйста|please)', ' ', q)
    core_words = len(re.findall(r'\w+', core))
    if not q:
        return None
    if re.search(r'(?:голов\w*|\bhead\b|шею|взгляд|камер\w*)', q):
        return None                      # "поверни голову": not the wheels
    driving = DRIVE.search(q) or LEG_CONTEXT.search(q) or (SPIN.search(q) and '?' not in q)
    if not driving and ('?' in q or core_words > 3):
        # Questions ("что у тебя впереди?") and longer talk never drive
        # without an explicit movement verb; a short "стоп" still stops.
        if not (LEG_STOP.search(q) and core_words <= 3):
            return None
    if LEG_STOP.search(q) and (driving or core_words <= 3):
        return _intent('stop')
    if FASTER.search(q) and (driving or 'скорост' in q or 'speed' in q or core_words <= 2):
        return _intent('faster')
    if SLOWER.search(q) and (driving or 'скорост' in q or 'speed' in q or core_words <= 2):
        return _intent('slower')
    seconds = SECONDS_RE.search(q)
    meters = METERS_RE.search(q)
    cm = CM_RE.search(q)
    degrees = DEGREES_RE.search(q)
    if not degrees and SPIN.search(q):
        # "развернись на 360", "крутись на 270": a bare angle after a turn word
        degrees = re.search(r'(?:на|by)\s+(\d{2,3})(?!\s*(?:\d|[.,]\d|секунд|сек|seconds?|метр|m\b|см|cm|сантиметр))', q)
    amount = {}
    if seconds:
        amount['seconds'] = min(MAX_SECONDS, _number(seconds.group(1)))
    elif meters:
        amount['meters'] = min(5.0, _number(meters.group(1)))
    elif cm:
        amount['meters'] = min(5.0, int(cm.group(1)) / 100.0)
    elif METER_WORD.search(q):
        amount['meters'] = 1.0
    elif A_BIT.search(q):
        amount['seconds'] = SHORT_SECONDS
    if SPIN.search(q) or (driving and (LEFT.search(q) or RIGHT.search(q)) and not FORWARD.search(q)
                          and not BACKWARD.search(q)):
        times = TIMES_RE.search(q)
        if times or FEW_TIMES.search(q):
            # "покрутись 3 раза", "несколько раз": N full turns (time limit applies)
            count = int(times.group(1)) if times else 3
            degrees_total = max(360, min(3600, 360 * count))
            if not (LEFT.search(q) or RIGHT.search(q)):
                return _intent('spin_right', degrees=degrees_total)
            return _intent('spin_left' if LEFT.search(q) else 'spin_right', degrees=degrees_total)
        if not (LEFT.search(q) or RIGHT.search(q)):
            if seconds:
                return _intent('spin_right', seconds=amount['seconds'])     # "покрутись 10 секунд"
            if AROUND.search(q) or degrees:
                # "развернись", "развернись на 360", "крутись на 270": no side said -> right
                turn = int(degrees.group(1)) if degrees else 180
                return _intent('spin_right', degrees=max(10, min(3600, turn)))
        if LEFT.search(q) or RIGHT.search(q):
            action = 'spin_left' if LEFT.search(q) and not RIGHT.search(q) else 'spin_right'
            if SPIN.search(q):
                turn = int(degrees.group(1)) if degrees else (180 if AROUND.search(q) else 90)
                if 'seconds' in amount and not degrees and seconds:
                    return _intent(action, seconds=amount['seconds'])
                return _intent(action, degrees=max(10, min(3600, turn)))
            # "езжай налево": drive in an arc
            return _intent('arc_left' if action == 'spin_left' else 'arc_right', **amount)
        return None
    if FORWARD.search(q) and not BACKWARD.search(q):
        if LEFT.search(q) and not RIGHT.search(q):
            return _intent('arc_left', **amount)
        if RIGHT.search(q) and not LEFT.search(q):
            return _intent('arc_right', **amount)
        return _intent('forward', **amount)
    if BACKWARD.search(q) and not FORWARD.search(q):
        return _intent('backward', **amount)
    if DRIVE.search(q) and amount and ('meters' in amount or seconds):
        return _intent('forward', **amount)       # "проедь 5 метров"
    return None


WHEE = {'Russian': 'Уииииии!', 'Kazakh': 'Уииииии!', 'Egyptian Arabic': 'ييييييه!', 'English': 'Wheeeee!'}


def is_whirl(intent):
    """Turning around or spinning: 180 degrees or more, or spinning for a time."""
    return (intent.get('action') in ('spin_left', 'spin_right')
            and (bool(intent.get('seconds')) or (intent.get('degrees') or 90) >= 180))


def legs_reply(intent, language='Russian', error=None):
    """Short spoken confirmation without a model call."""
    if language not in ('Russian', 'Kazakh', 'Egyptian Arabic'):
        language = 'English'
    if error is None and is_whirl(intent):
        return WHEE[language]               # she always shouts when asked to turn around
        return {'Russian': 'Не могу поехать: ' + str(error), 'Kazakh': 'Жүре алмаймын.',
                'Egyptian Arabic': 'مش قادرة أتحرك دلوقتي.',
                'English': 'I cannot drive right now.'}[language]
    words = {
        'Russian': {'forward': 'Еду вперёд.', 'backward': 'Еду назад.', 'arc_left': 'Еду налево.',
                    'arc_right': 'Еду направо.', 'spin_left': 'Поворачиваюсь налево.',
                    'spin_right': 'Поворачиваюсь направо.', 'stop': 'Стою.',
                    'faster': 'Быстрее.', 'slower': 'Медленнее.'},
        'Kazakh': {'forward': 'Алға жүремін.', 'backward': 'Артқа жүремін.', 'arc_left': 'Солға жүремін.',
                   'arc_right': 'Оңға жүремін.', 'spin_left': 'Солға бұрыламын.',
                   'spin_right': 'Оңға бұрыламын.', 'stop': 'Тоқтадым.',
                   'faster': 'Жылдамырақ.', 'slower': 'Баяуырақ.'},
        'Egyptian Arabic': {'forward': 'ماشية لقدام.', 'backward': 'راجعة لورا.', 'arc_left': 'ماشية شمال.',
                            'arc_right': 'ماشية يمين.', 'spin_left': 'بلف شمال.', 'spin_right': 'بلف يمين.',
                            'stop': 'وقفت.', 'faster': 'أسرع.', 'slower': 'أبطأ.'},
        'English': {'forward': 'Driving forward.', 'backward': 'Backing up.', 'arc_left': 'Driving left.',
                    'arc_right': 'Driving right.', 'spin_left': 'Turning left.', 'spin_right': 'Turning right.',
                    'stop': 'Stopped.', 'faster': 'Faster.', 'slower': 'Slower.'},
    }
    return words[language][intent['action']]
