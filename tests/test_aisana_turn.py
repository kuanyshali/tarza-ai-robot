"""Turn calibration with a synthetic 360-degree room (no camera, no wheels)."""
import os
import tempfile
import unittest

try:
    import cv2
    import numpy
except ImportError:                      # CI without OpenCV: the maths below still runs
    cv2 = numpy = None

import aisana_legs as legs
import aisana_turn as turn


class TableTests(unittest.TestCase):
    def test_seconds_for_an_angle(self):
        table = [[0.5, 20.0], [0.8, 45.0], [1.2, 90.0], [1.8, 170.0]]
        self.assertAlmostEqual(turn.seconds_for(90, table), 1.2)
        self.assertAlmostEqual(turn.seconds_for(45, table), 0.8)
        self.assertAlmostEqual(turn.seconds_for(130, table), 1.5)
        self.assertGreater(turn.seconds_for(360, table), 1.8)          # beyond: the last slope

    def test_legs_use_the_calibration(self):
        turns = {'table': [[0.5, 20.0], [0.8, 45.0], [1.2, 90.0], [1.8, 170.0]], 'speed_kmh': 2.0}
        spin = legs.make_intent('spin_right', degrees=90)
        self.assertAlmostEqual(legs.plan_seconds(spin, 2.0, turns=turns), 1.2)
        self.assertAlmostEqual(legs.plan_seconds(spin, 1.0, turns=turns), 2.4)   # half speed: twice the time
        self.assertAlmostEqual(legs.plan_seconds(spin, 2.0), legs.SPIN_90)       # no calibration: formula

    def test_saved_and_loaded(self):
        path = os.path.join(tempfile.mkdtemp(), 'turns.json')
        turn.save_calibration({'table': [[0.5, 20.0], [1.0, 60.0]], 'speed_kmh': 2.0}, path)
        self.assertEqual(turn.load_calibration(path)['table'][1], [1.0, 60.0])


@unittest.skipIf(cv2 is None, 'needs OpenCV')
class VisualGyroTests(unittest.TestCase):
    """The robot spins in a textured room; frames are windows of a 360-degree panorama."""

    def setUp(self):
        rng = numpy.random.RandomState(7)
        self.fov = 62.0
        self.width = turn.SMALL[0]
        self.room_width = int(round(self.width / self.fov * 360.0))     # pixels for 360 degrees
        noise = rng.rand(turn.SMALL[1] // 6, self.room_width // 6) * 255
        self.room = cv2.resize(noise.astype(numpy.uint8), (self.room_width, turn.SMALL[1]),
                               interpolation=cv2.INTER_CUBIC)
        self.angle = 0.0
        self.time = 0.0
        self.rate = 0.0                                                 # degrees per second now
        self.commands = []

    def picture(self):
        start = int(round(self.angle / 360.0 * self.room_width)) % self.room_width
        wide = numpy.concatenate([self.room, self.room], axis=1)
        return numpy.float32(wide[:, start:start + self.width])

    def clock(self):
        return self.time

    def next_picture(self):
        self.time += 0.1                                                # 10 frames per second
        self.angle += self.rate * 0.1
        return self.picture()

    def command(self, name):
        self.commands.append(name)
        # a real robot: spins right turn the view; 90 deg/s while the command lasts
        self.rate = {'spin_right': 90.0, 'spin_left': -90.0}.get(name, 0.0)
        return True

    def test_measures_turns_and_the_camera_view(self):
        log = []
        calibrator = turn.Calibrator(self.next_picture, self.command, log=log.append, clock=self.clock)
        data = calibrator.run()
        self.assertAlmostEqual(data['pixels_per_degree'], self.width / self.fov, delta=0.15)
        for seconds, degrees in data['table']:
            # 90 deg/s while commanded; the measurement must find that within a few degrees
            self.assertAlmostEqual(degrees, 90.0 * seconds, delta=6.0, msg=log)
        self.assertEqual(self.commands[-1], 'stop')

    def test_stop_ends_the_calibration(self):
        calibrator = turn.Calibrator(self.next_picture, self.command, log=lambda t: None,
                                     clock=self.clock, stopped=lambda: True)
        with self.assertRaises(RuntimeError):
            calibrator.run()
        self.assertEqual(self.commands[-1], 'stop')


if __name__ == '__main__':
    unittest.main()
