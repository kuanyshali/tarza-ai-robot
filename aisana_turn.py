"""Turn calibration with the camera ("visual gyroscope").

"Айсана, повернись на 90" under-turned (Nov 25): spins were timed by a formula.
Here she measures her own turns with the front camera:

1. one full turn: consecutive frames are matched by phase correlation (the
   horizontal shift in pixels is the turn); when the first picture comes back
   the shift so far is exactly 360 degrees, which gives pixels per degree
   without knowing the camera's field of view;
2. a few spins of different lengths, each measured the same way, give a table
   "seconds -> degrees" (it includes the ODrive ramp-up and the coasting);
3. aisana_legs plans spins from that table (saved in
   ~/.local/share/aisana/turn_calibration.json).

Needs a room with some texture (not a blank wall) and space to spin.
Python 3.6; cv2 and numpy from JetPack. Frames, clock and wheels are injected,
so the logic is tested without hardware.
"""
import json
import os
import time

CALIBRATION_FILE = os.environ.get('AISANA_TURN_FILE',
                                  os.path.expanduser('~/.local/share/aisana/turn_calibration.json'))
SMALL = (320, 180)
FALLBACK_FOV = 62.0                  # degrees, if the full turn is not recognised
TEST_SPINS = (0.5, 0.8, 1.2, 1.8)    # seconds of spin command for the table
COAST = 1.0                          # keep measuring after stop: the robot still turns a bit


def small_gray(frame):
    import cv2
    import numpy
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return numpy.float32(cv2.resize(gray, SMALL, interpolation=cv2.INTER_AREA))


class Shift(object):
    """Horizontal shift between two pictures (pixels of the small picture)."""

    def __init__(self):
        import cv2
        self.cv2 = cv2
        self.window = cv2.createHanningWindow(SMALL, cv2.CV_32F)

    def __call__(self, a, b):
        (dx, dy), response = self.cv2.phaseCorrelate(a, b, self.window)
        return dx, response


class YawTracker(object):
    """Sums the shifts of consecutive frames; recognises the first frame again."""

    MIN_RESPONSE = 0.05              # below this the match is noise (blur, blank wall)

    def __init__(self, shift=None):
        self.shift = shift or Shift()
        self.first = None
        self.last = None
        self.pixels = 0.0
        self.bad = 0
        self.frames = 0
        self.elapsed = 0.0

    def add(self, picture):
        if self.first is None:
            self.first = self.last = picture
            return 0.0
        dx, response = self.shift(self.last, picture)
        self.frames += 1
        if response < self.MIN_RESPONSE:
            self.bad += 1
        else:
            self.pixels += dx
        self.last = picture
        return dx

    def back_at_start(self, picture, min_pixels):
        """(True, pixels for 360) when the first picture is seen again after min_pixels."""
        if abs(self.pixels) < min_pixels:
            return False, None
        dx, response = self.shift(picture, self.first)
        if response < 0.25 or abs(dx) > SMALL[0] * 0.25:
            return False, None
        sign = 1.0 if self.pixels >= 0 else -1.0
        return True, abs(self.pixels + sign * abs(dx))


def seconds_for(degrees, table):
    """Spin time for an angle from a measured [[seconds, degrees], ...] table
    (linear between points, the last slope beyond them)."""
    points = sorted([(0.0, 0.0)] + [(float(s), float(d)) for s, d in table if d > 0])
    for (s1, d1), (s2, d2) in zip(points, points[1:]):
        if degrees <= d2 and d2 > d1:
            return s1 + (s2 - s1) * (degrees - d1) / (d2 - d1)
    (s1, d1), (s2, d2) = points[-2], points[-1]
    slope = (s2 - s1) / (d2 - d1) if d2 > d1 else 0.01
    return s2 + slope * (degrees - d2)


def load_calibration(path=None):
    try:
        with open(path or CALIBRATION_FILE) as handle:
            data = json.load(handle)
        if len(data.get('table') or []) >= 2:
            return data
    except (IOError, OSError, ValueError):
        pass
    return None


def save_calibration(data, path=None):
    path = path or CALIBRATION_FILE
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path + '.part', 'w') as handle:
        json.dump(data, handle, indent=1)
    os.rename(path + '.part', path)


class Calibrator(object):
    """Runs the calibration with injected camera / wheels / clock."""

    def __init__(self, next_picture, command, log=print, clock=time.monotonic, shift=None,
                 speed_kmh=2.0, stopped=lambda: False):
        self.next_picture = next_picture      # () -> small gray picture (a NEW frame) or None
        self.command = command                # (name) -> ok   'spin_right' / 'spin_left' / 'stop'
        self.log = log
        self.clock = clock
        self.shift = shift
        self.speed_kmh = speed_kmh
        self.stopped = stopped                # "стоп" was said

    def _measure(self, direction, seconds, tracker=None, until_start=False, min_pixels=0):
        tracker = tracker or YawTracker(self.shift)
        first = self.next_picture()
        if first is None:
            raise RuntimeError('no camera picture')
        if tracker.first is None:
            tracker.add(first)
        else:
            tracker.last = first             # continue a turn: the pause between drives is not a turn
        if not self.command(direction):
            raise RuntimeError('the wheels refused to spin (obstacle or motors off)')
        started = self.clock()
        end = started + seconds
        renew = self.clock() + 0.3
        closed = None
        try:
            while self.clock() < end and not self.stopped():
                if self.clock() >= renew:
                    self.command(direction)          # keeps the drive alive (lease) while it lasts
                    renew = self.clock() + 0.3
                picture = self.next_picture()
                if picture is None:
                    continue
                tracker.add(picture)
                if until_start:
                    found, full = tracker.back_at_start(picture, min_pixels)
                    if found:
                        closed = full
                        break
        finally:
            self.command('stop')
            tracker.elapsed = self.clock() - started    # the real spin time, not the planned one
        coast_end = self.clock() + COAST
        while self.clock() < coast_end and closed is None:
            picture = self.next_picture()
            if picture is not None:
                tracker.add(picture)
        return tracker, closed

    def run(self):
        # 1. full turn: pixels per 360 degrees
        guess = SMALL[0] / FALLBACK_FOV * 360.0
        tracker, full = self._measure('spin_right', 9.5, until_start=True, min_pixels=guess * 0.6)
        if full is None and not self.stopped():
            # Slow wheels: the turn is not done in one drive (<= 10 s each); go on.
            tracker, full = self._measure('spin_right', 9.5, tracker=tracker, until_start=True,
                                          min_pixels=guess * 0.6)
        if full and 0.6 * guess < full < 1.6 * guess:
            per_degree = full / 360.0
            self.log('Turn calibration: full turn = {:.0f} px, camera view {:.0f} degrees.'.format(
                full, SMALL[0] / per_degree))
        else:
            per_degree = SMALL[0] / FALLBACK_FOV
            self.log('Turn calibration: did not recognise the start again; assuming a {:.0f} degree '
                     'camera view.'.format(FALLBACK_FOV))
        # 2. spins of different lengths, alternating direction (she ends where she began)
        table = []
        direction = 'spin_left'
        for seconds in TEST_SPINS:
            if self.stopped():
                raise RuntimeError('stopped')
            tracker, _ = self._measure(direction, seconds)
            degrees = abs(tracker.pixels) / per_degree
            self.log('Turn calibration: {} {:.2f} s -> {:.0f} degrees ({} unclear frames).'.format(
                direction, tracker.elapsed, degrees, tracker.bad))
            table.append([round(tracker.elapsed, 2), round(degrees, 1)])
            direction = 'spin_right' if direction == 'spin_left' else 'spin_left'
        if not all(b[1] > a[1] for a, b in zip(table, table[1:])):
            raise RuntimeError('the measured turns do not grow with time (too little texture '
                               'for the camera?): {}'.format(table))
        return {'table': table, 'speed_kmh': self.speed_kmh, 'pixels_per_degree': per_degree,
                'when': time.strftime('%Y-%m-%d %H:%M')}
