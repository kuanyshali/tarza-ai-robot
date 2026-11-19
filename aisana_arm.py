"""USB serial link from Aisana (Jetson) to the Arm team's Arduino Mega.

The Arduino runs the Arm team's firmware (robotic_arms_*.ino, 115200 baud,
newline-terminated text commands such as "b 1", "l g o", "x", "p").
This module only ever sends commands built from a fixed whitelist, never
free text from speech recognition or from the language model.

Python 3.6 compatible. Needs pyserial on the Jetson:
    python3 -m pip install --user "pyserial==3.5"
and the user must be in the dialout group (sudo usermod -aG dialout $USER).

Environment:
    AISANA_ARM=0             disable the arm link completely (default 1)
    AISANA_ARM_PORT=/dev/... use this serial port instead of auto-detection
    AISANA_ARM_ECHO=0        do not print Arduino output to the console
"""
import collections
import os
import re
import threading
import time

BAUD = 115200
READY_TEXT = 'Both arms ready'
ARM_LETTERS = {'right': 'r', 'left': 'l', 'both': 'b'}
JOINT_LETTERS = {'whole': '', 'shoulder': 'm', 'elbow': 's'}
ACTIONS = ('position', 'gripper_open', 'gripper_close', 'stop', 'status')
# USB vendor IDs of genuine Arduino boards and common clone USB-serial chips.
ARDUINO_VIDS = (0x2341, 0x2A03, 0x1A86, 0x0403, 0x10C4)


class ArmError(RuntimeError):
    """The arm link is missing, not ready, or a command is not allowed."""


def build_command(action, arm='both', position=None, joint='whole'):
    """Translate a validated intent into one firmware command line."""
    if action == 'stop':
        return 'x'                       # firmware EMERGENCY STOP for everything
    if action == 'status':
        return 'p'
    letter = ARM_LETTERS.get(arm)
    if letter is None:
        raise ArmError('Unknown arm: {!r}'.format(arm))
    if action == 'position':
        try:
            number = int(position)
        except (TypeError, ValueError):
            raise ArmError('Position must be 0..4, got {!r}'.format(position))
        if number < 0 or number > 4 or str(number) != str(position).strip():
            raise ArmError('Position must be 0..4, got {!r}'.format(position))
        joint_letter = JOINT_LETTERS.get(joint or 'whole')
        if joint_letter is None:
            raise ArmError('Unknown joint: {!r}'.format(joint))
        return ' '.join(part for part in (letter, joint_letter, str(number)) if part)
    if action == 'gripper_open':
        return letter + ' g o'
    if action == 'gripper_close':
        return letter + ' g c'
    raise ArmError('Unknown arm action: {!r}'.format(action))


def find_arduino_port(ports=None, environ=None):
    """Return exactly one Arduino serial port; never guess between several."""
    environ = os.environ if environ is None else environ
    override = environ.get('AISANA_ARM_PORT', '').strip()
    if override:
        return override
    if ports is None:
        from serial.tools import list_ports
        ports = list_ports.comports()
    matches = sorted(set(
        port.device for port in ports
        if getattr(port, 'vid', None) in ARDUINO_VIDS
        or 'ttyACM' in (getattr(port, 'device', '') or '')
    ))
    if len(matches) != 1:
        raise ArmError(
            'Expected one Arduino serial port, found {}: {}. '
            'Plug in the Arduino Mega or set AISANA_ARM_PORT=/dev/ttyACM0.'.format(
                len(matches), ', '.join(matches) or 'none'))
    return matches[0]


class ArmController(object):
    """Owns the serial port; a background thread reads Arduino output."""

    def __init__(self, port, serial_factory=None, ready_timeout=15.0,
                 echo=True, log=print):
        self.port = port
        self._factory = serial_factory
        self._serial = None
        self._write_lock = threading.Lock()
        self._lines = collections.deque(maxlen=300)
        self._lines_lock = threading.Lock()
        self._sequence = 0
        self.ready = threading.Event()
        self._stop = threading.Event()
        self._reader = None
        self.ready_timeout = ready_timeout
        self.echo = echo
        self.log = log

    def open(self):
        factory = self._factory
        if factory is None:
            import serial

            def factory(port):
                return serial.Serial(port, BAUD, timeout=0.2, write_timeout=1)
        self._serial = factory(self.port)
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        # Opening the USB port restarts an Arduino Mega (DTR). The firmware is
        # designed for that: elbows start from saved calibration, steppers hold.
        self.log('Arm: connected to {}; waiting for "{}"...'.format(self.port, READY_TEXT))
        return self

    def close(self):
        self._stop.set()
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass

    def _read_loop(self):
        pending = b''
        while not self._stop.is_set():
            try:
                chunk = self._serial.read(256)
            except Exception as error:
                if not self._stop.is_set():
                    self.log('Arm: serial read failed: {}'.format(error))
                self.ready.clear()
                return
            if not chunk:
                continue
            pending += chunk
            while b'\n' in pending:
                raw, pending = pending.split(b'\n', 1)
                self._remember(raw.decode('utf-8', 'replace').strip())

    def _remember(self, line):
        if not line:
            return
        with self._lines_lock:
            self._sequence += 1
            self._lines.append((self._sequence, line))
        if self.echo:
            self.log('Arduino: ' + line)
        if READY_TEXT in line and not self.ready.is_set():
            self.ready.set()
            self.log('Arm: ready.')

    def _lines_after(self, sequence):
        with self._lines_lock:
            return [line for number, line in self._lines if number > sequence]

    def send(self, command, collect_seconds=0.0):
        """Write one whitelisted command; optionally collect the reply lines."""
        if self._serial is None:
            raise ArmError('Arm is not connected')
        if command != 'x' and not self.ready.wait(self.ready_timeout):
            raise ArmError('Arduino did not report "{}"'.format(READY_TEXT))
        with self._lines_lock:
            start = self._sequence
        with self._write_lock:
            self._serial.write((command + '\n').encode('ascii'))
            self._serial.flush()
        self.log('Arm command: ' + command)
        if collect_seconds > 0:
            time.sleep(collect_seconds)
        return self._lines_after(start)

    def execute(self, action, arm='both', position=None, joint='whole'):
        """Run an intent; returns a short English summary for the model/logs."""
        command = build_command(action, arm, position, joint)
        lines = self.send(command, collect_seconds=0.6 if action == 'status' else 0.15)
        problems = [line for line in lines if re.search(
            r'STOPPED|not set|OFF|limit|can\'t|cannot', line, re.I)]
        summary = 'Sent "{}" to the Arduino.'.format(command)
        if action == 'status':
            summary += ' Status: ' + ' | '.join(lines[-6:]) if lines else ' No status reply yet.'
        elif problems:
            summary += ' Arduino reported: ' + ' | '.join(problems[-3:])
        return summary

    def emergency_stop(self):
        try:
            self.send('x')
        except Exception as error:
            self.log('Arm: emergency stop could not be sent: {}'.format(error))


def connect_from_environment(log=print, environ=None):
    """Open the arm link if enabled and present; never stop the robot if not."""
    environ = os.environ if environ is None else environ
    if environ.get('AISANA_ARM', '1') != '1':
        log('Arm: disabled (AISANA_ARM=0).')
        return None
    try:
        port = find_arduino_port(environ=environ)
        controller = ArmController(port, echo=environ.get('AISANA_ARM_ECHO', '1') == '1', log=log)
        return controller.open()
    except ImportError:
        log('Arm: pyserial is missing; run python3 -m pip install --user "pyserial==3.5".')
    except Exception as error:
        log('Arm: not connected ({}). Voice and camera keep working.'.format(error))
    return None


# ---------------------------------------------------------------------------
# Local (no API call) parsing of explicit spoken arm commands, used by robot.py.
# A command needs an arm word AND an action, so "что у меня в руке?" is not one.
ARM_WORDS = re.compile(
    r'(?:рук[аиуеой]\w*|клешн\w*|захват\w*|кист[ьи]\w*|манипулятор\w*|'
    r'\barms?\b|\bgrippers?\b|\bhands?\b|\bclaws?\b|\bқол\w*|ذراع\w*|إيد\w*|ايد\w*|'
    r'локт\w*|локоть|плеч[оиа]\w*|\belbows?\b|\bshoulders?\b)', re.I)
NUMBER_WORDS = (
    (0, r'0|ноль|нуль|нулев\w*|zero|нөл\w*|صفر'),
    (1, r'1|один|одн[уа]|перв\w*|one|first|бір\w*|واحد'),
    (2, r'2|два|две|втор\w*|two|second|екі\w*|اتنين|اثنين'),
    (3, r'3|три|трет\w*|three|third|үш\w*|تلاتة|ثلاثة'),
    (4, r'4|четыре|четв[её]рт\w*|four|fourth|төрт\w*|اربعة|أربعة'),
)
POSITION_WORD = r'(?:позици\w*|положени\w*|positions?|pos|позиция\w*|қалып\w*|وضع\w*)'
HOME_WORDS = re.compile(
    r'(?:исходн\w*|в ноль|на ноль|домой|\bhome\b|\breset\b|бастапқы\w*)', re.I)
STOP_WORDS = re.compile(r'(?:\bстоп\b|останов\w*|замри|\bstop\b|\bhalt\b|freeze|тоқта\w*|وقف)', re.I)
OPEN_WORDS = re.compile(
    r'(?:открой|открыть|раскрой|разожми|разжать|отпусти|\bopen\b|\brelease\b|\bаш\b|افتح)', re.I)
CLOSE_WORDS = re.compile(
    r'(?:закрой|закрыть|сожми|сжать|схвати|возьми|\bclose\b|\bgrab\b|\bgrip\b|\bжап\b|اقفل|امسك)', re.I)
RIGHT_WORDS = re.compile(r'(?:прав\w*|\bright\b|\bоң\w*|يمين|اليمين)', re.I)
LEFT_WORDS = re.compile(r'(?:лев\w*|\bleft\b|\bсол\b|شمال|يسار|اليسار)', re.I)
SHOULDER_WORDS = re.compile(r'(?:плеч\w*|shoulder|stepper|шаговик\w*|иық\w*)', re.I)
ELBOW_WORDS = re.compile(r'(?:локт\w*|локоть|elbow|шынтақ\w*|كوع)', re.I)


def _position_number(text):
    for number, pattern in NUMBER_WORDS:
        if re.search(POSITION_WORD + r'\s*(?:номер\s*|number\s*|№\s*)?(?:' + pattern + r')\b', text, re.I):
            return number
        if re.search(r'\b(?:' + pattern + r')\s*' + POSITION_WORD, text, re.I):
            return number
    if HOME_WORDS.search(text):
        return 0
    return None


def parse_arm_command(text):
    """Return {'action', 'arm', 'position', 'joint'} or None if not an arm command."""
    q = ' '.join((text or '').casefold().split())
    if not q or not ARM_WORDS.search(q):
        return None
    if RIGHT_WORDS.search(q) and not LEFT_WORDS.search(q):
        arm = 'right'
    elif LEFT_WORDS.search(q) and not RIGHT_WORDS.search(q):
        arm = 'left'
    else:
        arm = 'both'                    # "руки", "обе руки", or both sides named
    if STOP_WORDS.search(q):
        return {'action': 'stop', 'arm': 'both', 'position': None, 'joint': 'whole'}
    number = _position_number(q)
    if number is not None:
        joint = 'whole'
        if SHOULDER_WORDS.search(q) and not ELBOW_WORDS.search(q):
            joint = 'shoulder'
        elif ELBOW_WORDS.search(q) and not SHOULDER_WORDS.search(q):
            joint = 'elbow'
        return {'action': 'position', 'arm': arm, 'position': number, 'joint': joint}
    if OPEN_WORDS.search(q):
        return {'action': 'gripper_open', 'arm': arm, 'position': None, 'joint': 'whole'}
    if CLOSE_WORDS.search(q):
        return {'action': 'gripper_close', 'arm': arm, 'position': None, 'joint': 'whole'}
    return None


_SIDE = {
    'Russian': {'right': 'правую руку', 'left': 'левую руку', 'both': 'обе руки'},
    'Kazakh': {'right': 'оң қолды', 'left': 'сол қолды', 'both': 'екі қолды'},
    'Egyptian Arabic': {'right': 'الإيد اليمين', 'left': 'الإيد الشمال', 'both': 'الإيدين'},
    'English': {'right': 'the right arm', 'left': 'the left arm', 'both': 'both arms'},
}


def arm_reply(intent, language='Russian', error=None):
    """Short spoken confirmation without an extra model call."""
    language = language if language in _SIDE else 'English'
    if error is not None:
        return {
            'Russian': 'Не могу управлять руками: контроллер рук не отвечает.',
            'Kazakh': 'Қолдарды басқара алмаймын: контроллер жауап бермей тұр.',
            'Egyptian Arabic': 'مش قادرة أحرك الإيدين، الكنترولر مش بيرد.',
            'English': 'I cannot move my arms right now: the arm controller is not responding.',
        }[language]
    side = _SIDE[language][intent.get('arm', 'both')]
    action = intent['action']
    if action == 'stop':
        return {'Russian': 'Остановила руки.', 'Kazakh': 'Қолдарды тоқтаттым.',
                'Egyptian Arabic': 'وقفت الإيدين.', 'English': 'Arms stopped.'}[language]
    if action == 'status':
        return {'Russian': 'Статус рук выведен в консоль.', 'Kazakh': 'Қолдардың күйі консольде.',
                'Egyptian Arabic': 'حالة الإيدين في الكونسول.', 'English': 'Arm status is in the console.'}[language]
    if action == 'position':
        number = intent['position']
        return {
            'Russian': 'Двигаю {} в позицию {}.'.format(side, number),
            'Kazakh': '{} {}-қалыпқа апарамын.'.format(side.capitalize(), number),
            'Egyptian Arabic': 'بحرك {} للوضع {}.'.format(side, number),
            'English': 'Moving {} to position {}.'.format(side, number),
        }[language]
    opening = action == 'gripper_open'
    return {
        'Russian': ('Открываю захват' if opening else 'Закрываю захват') + (' на обеих руках.' if intent.get('arm') == 'both' else ' на ' + {'right': 'правой', 'left': 'левой'}[intent['arm']] + ' руке.'),
        'Kazakh': side.capitalize() + (' ашамын.' if opening else ' жабамын.'),
        'Egyptian Arabic': ('بفتح ' if opening else 'بقفل ') + side + '.',
        'English': ('Opening' if opening else 'Closing') + ' the gripper on ' + side + '.',
    }[language]
