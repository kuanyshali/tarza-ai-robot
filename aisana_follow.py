"""'Айсана, иди за мной' / 'follow me': follow a person with the front camera.

The Jetson finds people with MobileNet-SSD (Caffe, PASCAL VOC, class 15 =
person) through cv2.dnn of JetPack's OpenCV 4.1.1: about 3 frames/s on the
Nano CPU while Aisana runs. Each step looks at the newest camera frame:

    person far to the left/right   -> a short spin toward them, then stop
    person a bit to the side, far  -> arc toward them
    person in front, far           -> forward
    person close (or obstacle)     -> stand still and wait

Safety (on top of the Pi's own interlocks):
- every command is re-sent each step; if this loop stops, the drive lease runs
  out and the keep-alive stops the wheels (aisana_legs.set_command);
- the Pi refuses everything but stop/backward when something is < 40 cm (409):
  she just waits;
- ends by itself after FOLLOW_SECONDS (default 60 s) or when nobody has been
  seen for LOST_SECONDS (3 s); "стоп" ends it at once.

Python 3.6, no hardware needed for the tests (detector, frames and legs are
injected).
"""
import hashlib
import os
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get('AISANA_MODEL_DIR', os.path.join(HERE, 'models'))
BASE_URL = 'https://github.com/chuanqi305/MobileNet-SSD/raw/master/'
# file name here -> (file in the chuanqi305/MobileNet-SSD repo, sha256)
MODEL_FILES = {
    'mobilenet_ssd_deploy.prototxt': (
        'deploy.prototxt', '2d180f723b3109e21f8287f6b3c691390d07b60eed998327cd3259ffa0e50608'),
    'mobilenet_iter_73000.caffemodel': (
        'mobilenet_iter_73000.caffemodel', '52eed8be80522c152a17fb56740de705b79881bde1a167e0e747310523685fc7'),
}
PERSON = 15                      # PASCAL VOC class id
MIN_SCORE = 0.5

FOLLOW_SECONDS = float(os.environ.get('AISANA_FOLLOW_SECONDS', '60'))
LOST_SECONDS = 3.0
# Distance from the box WIDTH (a standing person at 1-2 m does not fit the frame
# in height): ~0.30 of the frame width is about 1.5 m with the 62 degree camera.
NEAR_WIDTH = float(os.environ.get('AISANA_FOLLOW_NEAR', '0.30'))
FAR_WIDTH = NEAR_WIDTH * 0.8     # hysteresis: start walking again a bit further away
TURN_ZONE = 0.15                 # |offset from centre| (frame = -0.5..0.5) that needs a spin
ARC_ZONE = 0.07
CAMERA_FOV = 62.0                # degrees, horizontal


def ensure_model(log=print):
    """Paths (prototxt, caffemodel); downloads once with a sha256 check."""
    import requests
    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    paths = []
    for name in ('mobilenet_ssd_deploy.prototxt', 'mobilenet_iter_73000.caffemodel'):
        remote, digest = MODEL_FILES[name]
        path = os.path.join(MODEL_DIR, name)
        if not os.path.isfile(path) or _sha256(path) != digest:
            log('Follow: downloading the person detector ({})...'.format(name))
            data = requests.get(BASE_URL + remote, timeout=120).content
            if hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError('person detector download is corrupt ({})'.format(name))
            with open(path + '.part', 'wb') as handle:
                handle.write(data)
            os.rename(path + '.part', path)
        paths.append(path)
    return paths[0], paths[1]


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


class PersonDetector(object):
    def __init__(self, prototxt, caffemodel):
        import cv2
        self.cv2 = cv2
        self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

    def detect(self, frame):
        """[(x1, y1, x2, y2, score)] of people, coordinates 0..1."""
        cv2 = self.cv2
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        out = self.net.forward()
        people = []
        for row in out[0, 0]:
            if int(row[1]) == PERSON and row[2] >= MIN_SCORE:
                x1, y1, x2, y2 = [min(1.0, max(0.0, float(v))) for v in row[3:7]]
                if x2 > x1 and y2 > y1:
                    people.append((x1, y1, x2, y2, float(row[2])))
        return people


def centre(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def choose_target(people, previous):
    """Keep following the same person: the box nearest to the last one; the
    biggest (closest) person when starting or when the old one is gone."""
    if not people:
        return None
    if previous is not None:
        px, py = centre(previous)
        best = min(people, key=lambda b: (centre(b)[0] - px) ** 2 + (centre(b)[1] - py) ** 2)
        bx, by = centre(best)
        if (bx - px) ** 2 + (by - py) ** 2 < 0.3 ** 2:
            return best
    return max(people, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def decide(box, walking, degrees_per_second=107.0):
    """(command, spin_seconds or None, walking). Spins are short pulses sized to the
    angle; forward/arc last until the next step."""
    offset = centre(box)[0] - 0.5            # < 0: person on her left
    width = box[2] - box[0]
    if abs(offset) > TURN_ZONE:
        angle = abs(offset) * CAMERA_FOV
        pulse = min(0.35, max(0.08, angle / degrees_per_second))
        return ('spin_left' if offset < 0 else 'spin_right'), pulse, walking
    limit = FAR_WIDTH if not walking else NEAR_WIDTH
    if width < limit:
        if abs(offset) > ARC_ZONE:
            return ('arc_left' if offset < 0 else 'arc_right'), None, True
        return 'forward', None, True
    return 'stop', None, False


class Follower(object):
    """One follow run in a background thread."""

    def __init__(self, legs, get_frame, detector, log=print, on_end=None,
                 seconds=FOLLOW_SECONDS, lost_seconds=LOST_SECONDS, sleep=time.sleep):
        self.legs = legs
        self.get_frame = get_frame
        self.detector = detector
        self.log = log
        self.on_end = on_end or (lambda reason: None)
        self.seconds = seconds
        self.lost_seconds = lost_seconds
        self.sleep = sleep
        self.cancelled = threading.Event()
        self.thread = None
        self.reason = None

    def active(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def cancel(self):
        self.cancelled.set()

    def _command(self, command):
        try:
            code, data = self.legs.set_command(command)
        except Exception as error:
            self.log('Follow: drive command failed ({}).'.format(error))
            return False
        if code == 409:
            return False                      # something closer than 40 cm: wait
        if code != 200:
            self.log('Follow: the Pi refused {} ({}).'.format(command, data.get('error') or code))
            return False
        return True

    def run(self):
        reason = 'stopped'
        degrees = 90.0 / max(0.1, getattr(self.legs, 'spin_90', 0.84))
        try:
            self.legs.ensure_enabled()
            end = time.monotonic() + self.seconds
            last_seen = time.monotonic()
            target, walking, last = None, False, None
            while not self.cancelled.is_set():
                now = time.monotonic()
                if now > end:
                    reason = 'time'
                    break
                frame = self.get_frame()
                people = self.detector.detect(frame) if frame is not None else []
                target = choose_target(people, target)
                if target is None:
                    if last != 'stop':
                        self._command('stop')
                        last = 'stop'
                    walking = False
                    if now - last_seen > self.lost_seconds:
                        reason = 'lost'
                        break
                    self.sleep(0.1)
                    continue
                last_seen = now
                command, pulse, walking = decide(target, walking, degrees)
                if command != last or command != 'stop':
                    self._command(command)       # re-sent every step: renews the drive lease
                last = command
                if pulse:
                    self.sleep(pulse)
                    self._command('stop')
                    last = 'stop'
        except Exception as error:
            reason = 'error: {}'.format(error)
        finally:
            try:
                self.legs.set_command('stop')
            except Exception:
                pass
            self.reason = reason
            self.log('Follow: ended ({}).'.format(reason))
            self.on_end(reason)
