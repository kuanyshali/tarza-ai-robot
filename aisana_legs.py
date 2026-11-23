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
- every drive has a time limit (default 2 s, max 10 s);
- the Pi stops motion 0.35 s and switches the motors OFF 0.75 s after the last
  heartbeat (odrive_controller.py LINK_TIMEOUT_S), so if the Jetson, the LAN
  or this program dies the wheels stop by themselves;
- the Pi's obstacle stop (< 40 cm) answers 409 and the drive ends;
- motors are never calibrated by voice.

Keep-alive (Nov 23, owner: "everything automatic"): like the Pi's own /control
page, a background thread sends a heartbeat every 0.15 s while the motors are
on, so they stay on between commands (before, they switched off 0.75 s after
each drive: "the wheels only work while the /control page is open"). If they
are off it enables them again without calibration (nothing moves), and if the
ODrive needs calibration it calibrates on its own (the wheels spin a little,
announced first; owner's choice), at most twice per 10 minutes.

Python 3.6 compatible. Environment:
    AISANA_LEGS_URL=http://192.168.1.50:5000   the Pi (unset = legs disabled)
    AISANA_LEGS_SPIN_90_SECONDS=0.84           spin time for a 90 degree turn (360 = 2x)
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
HEARTBEAT_SECONDS = 0.08          # fixed rate; the Pi stops after 0.35 s without one
KEEPALIVE_SECONDS = 0.15         # like the Pi's /control page (150 ms)
AUTO_CALIBRATIONS = 2             # at most this many automatic calibrations ...
AUTO_CALIBRATION_WINDOW = 600.0   # ... per 10 minutes (a broken encoder must not loop)
MAX_RESUMES = 5                   # silent restarts after a Pi stop that was NOT an obstacle
ENABLE_TIMEOUT = 12.0             # odrive.find_any() can take a while the first time


class LegsError(RuntimeError):
    """The Pi is unreachable, motors cannot be enabled, or a command is refused."""


# Nov 23: turns were too short; all turns 1.4x longer (was 0.6 s per 90 degrees).
SPIN_90 = 0.84


def spin_90_seconds(environ=None):
    environ = os.environ if environ is None else environ
    try:
        return max(0.2, min(5.0, float(environ.get('AISANA_LEGS_SPIN_90_SECONDS', str(SPIN_90)))))
    except ValueError:
        return SPIN_90


def _intent(action, seconds=None, meters=None, degrees=None):
    return {'action': action, 'seconds': seconds, 'meters': meters, 'degrees': degrees}


def plan_seconds(intent, speed_kmh=2.0, spin_90=SPIN_90):
    """How long to drive for an intent; always within (0, MAX_SECONDS]."""
    if intent.get('seconds'):
        seconds = float(intent['seconds'])
    elif intent.get('meters'):
        seconds = float(intent['meters']) / max(0.1, float(speed_kmh) / 3.6)
    elif intent['action'] in ('spin_left', 'spin_right'):
        # The ODrive is still ramping up during short turns, so the angle grows
        # with the square of the time: 90 deg = spin_90, 360 deg = 2 x spin_90.
        # (Robot, Nov 19: 1.5 s gave about one and a half turns.)
        degrees = float(intent.get('degrees') or 90)
        if degrees <= 360:
            seconds = spin_90 * (degrees / 90.0) ** 0.5
        else:
            seconds = 2 * spin_90 * degrees / 360.0
    else:
        seconds = DEFAULT_SECONDS
    return max(0.3, min(MAX_SECONDS, seconds))


def short_error(error):
    """'timed out' instead of three lines of urllib3 (Jetson log, Nov 23)."""
    text = str(error)
    if 'timed out' in text or 'Timeout' in text:
        return 'no answer: Pi off, cable out or another IP'
    if 'Connection refused' in text:
        return 'Pi answers but the legs app (port 5000) is not running'
    if 'Name or service' in text or 'No route' in text or 'unreachable' in text:
        return 'no route to the Pi'
    return text[:120]


class LegsController(object):
    """HTTP client for the Pi plus one background drive at a time."""

    def __init__(self, base_url, http=None, log=print, spin_90=SPIN_90):
        if http is None:
            import requests
            http = requests.Session()        # keep-alive: no new TCP connection per heartbeat
        self.base_url = base_url.rstrip('/')
        self.http = http
        self.log = log
        self.spin_90 = spin_90
        self._cancel = threading.Event()
        self._drive = None
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._keepalive = None
        self.auto_calibrate = True
        self.on_calibrate = None             # called before an automatic calibration (Aisana warns)
        self.calibrations = []               # times of automatic calibrations
        self.alive_on = False                # motors on and kept on by the keep-alive
        self._next_check = 0.0
        self._enabling_until = 0.0           # just asked to enable: poll fast, heartbeat at once
        self.motion_lease_until = 0.0        # a drive/follow loop owns the motion until then
        self._last_note = ''

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

    # --- keep-alive ----------------------------------------------------------
    def start_keepalive(self, auto_calibrate=True):
        if self._keepalive is not None:
            return self
        self.auto_calibrate = auto_calibrate
        self._keepalive = threading.Thread(target=self._keepalive_loop, daemon=True)
        self._keepalive.start()
        return self

    def close(self):
        self._closed.set()

    def _note(self, text):
        if text != self._last_note:          # each new situation once, not every 3 s
            self._last_note = text
            self.log('Legs: ' + text)

    def _keepalive_loop(self):
        while not self._closed.is_set():
            try:
                wait = self.keepalive_step()
            except Exception as error:
                self.alive_on = False
                self._note('Pi not reachable ({}); retrying.'.format(short_error(error)))
                wait = 3.0
            self._closed.wait(wait)

    def keepalive_step(self, now=None):
        """One step; returns seconds to wait. Never moves the wheels except for
        an automatic calibration."""
        now = time.monotonic() if now is None else now
        if self.alive_on:
            code, data = self._post('/motor/heartbeat', timeout=0.3)
            if data.get('enabled'):
                if data.get('command', 'stop') != 'stop' and time.monotonic() > self.motion_lease_until:
                    # The keep-alive must never keep a drive going that nobody runs any
                    # more (its loop died or hung): without it the Pi would stop after 0.35 s.
                    self._post('/motor/command', {'command': 'stop'}, timeout=0.5)
                    self._note('stopped a drive with no owner ({}).'.format(data.get('command')))
                return KEEPALIVE_SECONDS
            self.alive_on = False
            self._next_check = 0.0
            self._note('motors went off ({}); switching them on again.'.format(data.get('state', '?')))
        if now < self._next_check:
            return 0.2
        # Right after an enable request poll every 0.2 s: the Pi switches the motors
        # off again 0.75 s after enabling unless a heartbeat comes (Nov 23 log:
        # "switching the motors on" -> CONTROL LINK LOST -> on -> lost ...).
        self._next_check = now + (0.2 if now < self._enabling_until else 3.0)
        status = self.status()
        if status.get('enabled'):
            self.alive_on = True
            self._note('motors on; keeping them on.')
            return KEEPALIVE_SECONDS
        if status.get('connecting'):
            self._note('the Pi is connecting to the ODrive ({}).'.format(status.get('state', '?')))
            self._enabling_until = max(self._enabling_until, now + 2.0)
            self._next_check = now + 0.2
            return 0.2
        state = str(status.get('state', ''))
        error = str(status.get('error') or '')
        if 'NEEDS CALIBRATION' in state or (state == 'ERROR' and 'calibration failed' in error):
            if not self.auto_calibrate:
                self._note('motors need calibration: press "Колёса" on the panel.')
                return 0.5
            self.calibrations = [t for t in self.calibrations if now - t < AUTO_CALIBRATION_WINDOW]
            if len(self.calibrations) >= AUTO_CALIBRATIONS:
                self._note('calibration keeps failing ({}); check the encoder wiring. '
                           'Press "Колёса" on the panel to try again.'.format(error or state))
                return 0.5
            self.calibrations.append(now)
            self._note('calibrating the wheels (they spin a little).')
            if self.on_calibrate is not None:
                try:
                    self.on_calibrate()
                except Exception:
                    pass
            self._closed.wait(3.0)            # time to step away
            self._post('/motor/enable', {'calibrate': True}, timeout=2.0)
            self._enabling_until = now + 60.0  # calibration, then closed loop
            self._next_check = now + 0.2
            return 0.2
        # DISCONNECTED, CONTROL LINK LOST, DISABLED or another error: switch on
        # without calibration (keeps the old calibration, nothing moves).
        self._note('switching the motors on ({}).'.format(state or '?'))
        self._post('/motor/enable', {'calibrate': False}, timeout=2.0)
        self._enabling_until = now + ENABLE_TIMEOUT
        self._next_check = now + (0.2 if state != 'ERROR' else 15.0)
        return 0.2

    # --- drives ------------------------------------------------------------
    def obstacle(self):
        """'obstacle 29 cm' style text if the Pi's ultrasonic sensors see one, else ''."""
        try:
            data = self.http.get(self.base_url + '/sensor-data', timeout=0.5).json()
        except Exception:
            return ''
        if not data.get('obstacle_stop'):
            return ''
        distances = [d for d in (data.get('distance_1'), data.get('distance_2')) if d is not None]
        return '{} cm away'.format(int(min(distances))) if distances else 'distance unknown'

    def set_command(self, command, lease=0.9):
        """One drive command for a loop that re-sends it every step (follow me).
        The motion is kept alive only `lease` seconds past the last call."""
        if command not in DRIVE_COMMANDS + ('stop',):
            raise LegsError('Unknown drive: {!r}'.format(command))
        self.motion_lease_until = time.monotonic() + (lease if command != 'stop' else 0.0)
        return self._post('/motor/command', {'command': command}, timeout=0.5)

    def stop(self):
        """Stop now. Safe to call any time; never raises."""
        self.motion_lease_until = 0.0
        self._cancel.set()
        try:
            self._post('/motor/command', {'command': 'stop'}, timeout=0.5)
        except Exception as error:
            # No heartbeat any more: the Pi stops on its own within 0.35 s.
            self.log('Legs: stop request failed ({}); the Pi stops by itself.'.format(short_error(error)))

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
        self.motion_lease_until = time.monotonic() + 0.5
        code, data = self._post('/motor/command', {'command': action})
        if code != 200:
            raise LegsError(data.get('error') or 'command refused ({})'.format(code))
        self.log('Legs: {} for {:.1f} s'.format(action, seconds))

        def worker():
            resumes = 0
            try:
                end = time.monotonic() + seconds
                next_beat = time.monotonic()
                while time.monotonic() < end and not cancel.is_set():
                    self.motion_lease_until = time.monotonic() + 0.5
                    try:
                        code, data = self._post('/motor/heartbeat', timeout=0.25)
                        stopped = code != 200 or data.get('command') == 'stop'
                    except Exception as error:
                        self.log('Legs: heartbeat failed ({})'.format(error))
                        stopped = True
                    if stopped and not cancel.is_set():
                        obstacle = self.obstacle()
                        if obstacle:
                            raise LegsError('obstacle ahead: ' + obstacle)
                        if resumes >= MAX_RESUMES:
                            self.log('Legs: the Pi keeps stopping without an obstacle; drive ended.')
                            return
                        resumes += 1
                        # Not an obstacle (a late heartbeat): continue silently.
                        self.log('Legs: Pi stopped without an obstacle; resuming ({}).'.format(resumes))
                        self.ensure_enabled()
                        self._post('/motor/command', {'command': action})
                    # Fixed rate: the next heartbeat does not wait for a slow answer.
                    next_beat += HEARTBEAT_SECONDS
                    cancel.wait(max(0.0, next_beat - time.monotonic()))
            except Exception as error:
                if not cancel.is_set():
                    self.log('Legs problem: {}'.format(error))
                    if 'obstacle' in str(error):
                        report(str(error))       # only real obstacles are spoken
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
        log('Legs: Pi at {} not reachable ({}); will retry on each command.'.format(url, short_error(error)))
    if environ.get('AISANA_LEGS_KEEPALIVE', '1') == '1':
        legs.start_keepalive(auto_calibrate=environ.get('AISANA_LEGS_AUTO_CALIBRATE', '1') == '1')
    return legs


# ---------------------------------------------------------------------------
# Local parsing of spoken drive commands. A command needs a movement word;
# arm commands are handled first by aisana_arm, so "руку вперёд" never drives.
FORWARD = re.compile(r'(?:вп[её]р[её]д\b|\bforward\b|\bahead\b|алға|لقدام|قدام)', re.I)
BACKWARD = re.compile(r'(?:назад|\bback(?:wards?)?\b|артқа|لورا|ورا)', re.I)
LEFT = re.compile(r'(?:налево|влево|лево|левее|\bleft\b|солға|شمال|يسار)', re.I)
RIGHT = re.compile(r'(?:направо|вправо|право|правее|\bright\b|оңға|يمين)', re.I)
# "поведи/повели" is how STT often hears "поверни" (Jetson log, Nov 19).
SPIN = re.compile(r'(?:поверн\w*|повед[иь]\w*|повели\b|разверн\w*|развед[иь]с\w*|развел[иь]с\w*|(?:^|,\s*)разбер[иь]сь[.!]?\s*$|разверт\w*|крут\w*|кружи\w*|покружи\w*|закружи\w*|вращай\w*|'
                  r'разворот\w*|\bturn\b|\bspin\b|'
                  r'бұрыл\w*|لف)', re.I)
TIMES_RE = re.compile(r'(\d+)\s*(?:раз\w*|times|рет|مرات|مرة)', re.I)
FEW_TIMES = re.compile(r'(?:несколько раз|пару раз|много раз|a few times|several times|бірнеше рет)', re.I)
# "разведись", "развертись", "развелись", "<имя>, разберись" are how STT hears "развернись" (Jetson log, Nov 19).
AROUND = re.compile(r'(?:разверн\w*|развед[иь]с\w*|развел[иь]с\w*|(?:^|,\s*)разбер[иь]сь[.!]?\s*$|разверт\w*|разворот\w*|turn around|кругом)', re.I)
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
# "Подойди", "ближе", "come closer" (STT often writes "came closer", Nov 23 log).
APPROACH = re.compile(r'(?:подойди\w*|подходи\w*|подъезжай|иди сюда|иди ко мне|ко мне\b|\w*ближе\b|'
                      r'\bc[oa]me (?:closer|here|to me|over)\b|\bcloser\b|\bapproach\b|жақында\w*|бері кел\w*)', re.I)
BACK_OFF = re.compile(r'(?:отойди\w*|отъедь\w*|подальше|\bgo away\b|\bback off\b|\bfurther\b|алыстa\w*)', re.I)
LEG_CONTEXT = re.compile(r'(?:\bноги\b|колёс\w*|колес\w*|\blegs?\b|wheels?|робот\w*)', re.I)


make_intent = _intent


def mentions_driving(text):
    """Does the user talk about driving at all (for vague requests the model interprets)?"""
    q = (text or '').casefold()
    return bool(DRIVE.search(q) or LEG_CONTEXT.search(q) or FORWARD.search(q) or BACKWARD.search(q)
                or APPROACH.search(q) or BACK_OFF.search(q)
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
    # "... и крикни «We»": what to shout is not part of the drive command.
    q = re.sub(r'\s*(?:,|\bи\b|\band\b)?\s*(?:крикни|кричи|прокричи|покричи|заори|'
               r'shout|yell|scream|айқайла\w*)\b.*$', '', q)
    if re.search(r'(?:смотр\w*|видишь|\blook\b|\bsee\b|қара\w*)', q) and not DRIVE.search(q):
        return None                      # "смотри вперёд" is about the camera, not the wheels
    # The name and fillers do not count as words: "Айсана, вперёд!" is short.
    core = re.sub(r'(?:айсан\w*|эйсан\w*|aisan\w*|aysan\w*|давай|ну|пожалуйста|please)', ' ', q)
    core_words = len(re.findall(r'\w+', core))
    if not q:
        return None
    if re.search(r'(?:голов\w*|\bhead\b|шею|взгляд|камер\w*)', q):
        return None                      # "поверни голову": not the wheels
    driving = DRIVE.search(q) or LEG_CONTEXT.search(q) or (SPIN.search(q) and '?' not in q)
    if '?' not in q and core_words <= 6 and not re.search(r'(?:рук|руч|\barms?\b|\bhands?\b)', q):
        near, away = APPROACH.search(q), BACK_OFF.search(q)
        if near or away:
            # A short careful step toward / away from the speaker; the Pi's
            # obstacle stop (< 40 cm) ends it before it touches anyone.
            seconds, meters = SECONDS_RE.search(q), METERS_RE.search(q)
            intent = _intent('forward' if near else 'backward', seconds=SHORT_SECONDS)
            if seconds:
                intent['seconds'] = min(MAX_SECONDS, _number(seconds.group(1)))
            elif meters:
                intent['seconds'], intent['meters'] = None, min(5.0, _number(meters.group(1)))
            return intent
    if not driving and '?' not in q and core_words <= 6 and (FORWARD.search(q) or BACKWARD.search(q)) \
            and not re.search(r'(?:рук|руч|плеч|локот|локт|кист|ладон|\barms?\b|\bhands?\b|shoulder|elbow|расскаж|tell)', q) and (
            METERS_RE.search(q) or CM_RE.search(q) or SECONDS_RE.search(q)):
        # "Айсана, на один метр назад": a direction plus a distance or time is a drive
        # even without a verb (Nov 23: ignored, then "I backed up" with nothing moved).
        driving = True
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
            if re.search(r'(?:кружи|крут|вращай|\bspin\b)', q):
                return _intent('spin_right', degrees=360)      # "кружись", "покрутись": one full turn
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
