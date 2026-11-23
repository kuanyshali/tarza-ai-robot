"""Offline tests for voice driving; a fake Raspberry Pi stands in for app.py."""
import threading
import time
import unittest

import aisana_legs as legs


class FakeResponse(object):
    def __init__(self, code, data):
        self.status_code = code
        self._data = data

    def json(self):
        return self._data


class FakePi(object):
    """Mimics the legs team's Flask API (app.py) and its safety monitor."""

    def __init__(self, needs_calibration=False, obstacle_after=None):
        self.enabled = False
        self.command = 'stop'
        self.commands = []
        self.heartbeats = 0
        self.speed = 2.0
        self.needs_calibration = needs_calibration
        self.obstacle_after = obstacle_after
        self.started = None
        self.lock = threading.Lock()

    def status(self):
        state = 'ENABLED' if self.enabled else ('NEEDS CALIBRATION' if self.needs_calibration else 'DISABLED')
        return {'enabled': self.enabled, 'state': state, 'command': self.command,
                'speed_kmh': self.speed, 'error': None}

    def get(self, url, timeout=None):
        return FakeResponse(200, self.status())

    def post(self, url, json=None, timeout=None):
        path = url.split(':5000', 1)[1]
        with self.lock:
            if path == '/motor/enable':
                self.enabled = not self.needs_calibration and not json.get('calibrate')
                return FakeResponse(200, self.status())
            if path == '/motor/heartbeat':
                self.heartbeats += 1
                if (self.obstacle_after is not None and self.started
                        and time.monotonic() - self.started > self.obstacle_after):
                    self.command = 'stop'              # ultrasonic interlock in app.py
                return FakeResponse(200, self.status())
            if path == '/motor/speed':
                self.speed = max(0.5, min(5.5, self.speed + json.get('delta', 0)))
                return FakeResponse(200, self.status())
            if path == '/motor/command':
                if not self.enabled:
                    return FakeResponse(400, {'error': 'Motors are not enabled.'})
                self.command = json['command']
                self.commands.append(json['command'])
                if json['command'] != 'stop':
                    self.started = time.monotonic()
                return FakeResponse(200, self.status())
        return FakeResponse(404, {})


def controller(pi, problems=None):
    return legs.LegsController('http://10.0.0.2:5000', http=pi, log=lambda *a: None)


class ParserTests(unittest.TestCase):
    def check(self, text, action, seconds=None, meters=None, degrees=None):
        self.assertEqual(legs.parse_legs_command(text),
                         {'action': action, 'seconds': seconds, 'meters': meters, 'degrees': degrees}, text)

    def test_drive_commands(self):
        self.check('Айсана, езжай вперёд', 'forward')
        self.check('Айсана, вперёд!', 'forward')
        self.check('проедь вперёд 3 секунды', 'forward', seconds=3.0)
        self.check('проедь вперёд на метр', 'forward', meters=1.0)
        self.check('подъедь вперёд на 50 сантиметров', 'forward', meters=0.5)
        self.check('отъедь назад немного', 'backward', seconds=1.0)
        self.check('повернись налево', 'spin_left', degrees=90)
        self.check('повернись направо на 45 градусов', 'spin_right', degrees=45)
        self.check('Айсана, развернись', 'spin_right', degrees=180)
        self.check('езжай налево', 'arc_left')
        self.check('drive forward 2 seconds', 'forward', seconds=2.0)
        self.check('go forward one meter', 'forward', meters=1.0)
        self.check('turn left', 'spin_left', degrees=90)
        self.check('Айсана, стоп', 'stop')
        self.check('стой', 'stop')
        self.check('едь быстрее', 'faster')
        # From the Jetson log: number words and the noun "проезд".
        self.check('Проезд пять метров', 'forward', meters=5.0)
        self.check('проедь пять метров', 'forward', meters=5.0)
        self.check('проехать вперёд две секунды', 'forward', seconds=2.0)
        self.check('отъедь назад на полметра', 'backward', meters=0.5)
        self.check('Айсана, вперед 4 секунды', 'forward', seconds=4.0)
        self.check('Айсана, проейдь пять секунд', 'forward', seconds=5.0)
        # From the Jetson log: "поведи" heard instead of "поверни"; 360 was ignored.
        self.check('Айсана, поведи налево', 'spin_left', degrees=90)
        self.check('повели направо', 'spin_right', degrees=90)
        self.check('развернись на 360', 'spin_right', degrees=360)
        self.check('крутись на 270', 'spin_right', degrees=270)
        # Jetson log: "Асама, разведись" (= Айсана, развернись); shout tail must not break parsing.
        self.check('Асама, разведись.', 'spin_right', degrees=180)
        self.check('Вперёд 2 секунды и крикни «We».', 'forward', seconds=2.0)
        self.check('Айсана, развернись и кричи: «Вии!»', 'spin_right', degrees=180)
        self.check('поверни налево на 360 градусов', 'spin_left', degrees=360)

    def test_not_drive_commands(self):
        for text in ('повели меня в магазин', 'что у тебя впереди?', 'Айсана, смотри вперёд и скажи что видишь',
                     'вернись назад к теме разговора', 'поверни голову направо?',
                     'какая погода', 'правую руку в позицию 2', 'назад в будущее это фильм', ''):
            with self.subTest(text=text):
                self.assertIsNone(legs.parse_legs_command(text))

    def test_time_is_always_limited(self):
        self.assertEqual(legs.plan_seconds(legs.make_intent('forward', seconds=60)), legs.MAX_SECONDS)
        self.assertEqual(legs.plan_seconds(legs.make_intent('forward')), legs.DEFAULT_SECONDS)
        self.assertAlmostEqual(legs.plan_seconds(legs.make_intent('forward', meters=1.0), 3.6), 1.0)
        # Robot, Nov 19: 1.5 s spun ~1.5 turns while the ODrive was still ramping, so the
        # angle grows with time squared: 90 deg = 0.84 s, 360 deg = 1.68 s (1.4x since Nov 23).
        self.assertAlmostEqual(legs.plan_seconds(legs.make_intent('spin_left', degrees=90)), 0.84)
        self.assertAlmostEqual(legs.plan_seconds(legs.make_intent('spin_left', degrees=360)), 1.68)
        self.assertAlmostEqual(legs.plan_seconds(legs.make_intent('spin_left', degrees=720)), 3.36)


class ControllerTests(unittest.TestCase):
    def test_drive_enables_heartbeats_and_stops_by_itself(self):
        pi = FakePi()
        legs_ctrl = controller(pi)
        seconds = legs_ctrl.drive(legs.make_intent('forward', seconds=0.6))
        self.assertAlmostEqual(seconds, 0.6)
        legs_ctrl._drive.join(3)
        self.assertEqual(pi.commands, ['forward', 'stop'])
        self.assertGreaterEqual(pi.heartbeats, 3)          # every ~0.12 s, Pi needs < 0.35 s

    def test_voice_stop_cuts_the_drive_short(self):
        pi = FakePi()
        legs_ctrl = controller(pi)
        legs_ctrl.drive(legs.make_intent('forward', seconds=5))
        time.sleep(0.3)
        started = time.monotonic()
        legs_ctrl.stop()
        legs_ctrl._drive.join(2)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(pi.command, 'stop')

    def test_never_calibrates_by_voice(self):
        pi = FakePi(needs_calibration=True)
        problems = []
        legs_ctrl = controller(pi)
        with self.assertRaisesRegex(legs.LegsError, 'calibration'):
            legs_ctrl.drive(legs.make_intent('forward'), problems.append)
        self.assertEqual(pi.commands, [])

    def test_execute_says_not_driving_when_calibration_needed(self):
        pi = FakePi(needs_calibration=True)
        result = controller(pi).execute(legs.make_intent('forward'))
        self.assertTrue(result.startswith('NOT driving'))
        self.assertIn('calibration', result)
        self.assertEqual(pi.commands, [])

    def test_obstacle_refusal_known_before_reply(self):
        # From the Jetson log: Pi answered 409 but Aisana had already said "Еду вперёд".
        pi = FakePi()
        pi.enabled = True
        original = pi.post

        def blocked(url, json=None, timeout=None):
            if url.endswith('/motor/command') and json.get('command') != 'stop':
                return FakeResponse(409, {'error': 'Movement blocked: obstacle closer than 40 cm.'})
            return original(url, json, timeout)
        pi.post = blocked
        result = controller(pi).execute(legs.make_intent('forward'))
        self.assertTrue(result.startswith('NOT driving'))
        self.assertIn('obstacle', result)

    def test_obstacle_stop_is_reported(self):
        pi = FakePi(obstacle_after=0.2)
        pi.get = lambda url, timeout=None: FakeResponse(
            200, {'obstacle_stop': True, 'distance_1': 20.0} if 'sensor' in url else pi.status())
        problems = []
        legs_ctrl = controller(pi)
        legs_ctrl.drive(legs.make_intent('forward', seconds=3), problems.append)
        legs_ctrl._drive.join(3)
        self.assertIn('obstacle', problems[0])

    def test_late_heartbeat_stop_resumes_silently(self):
        # The Pi stops when a heartbeat is late; without an obstacle the drive continues.
        pi = FakePi()
        original = pi.post
        state = {'beats': 0}

        def flaky(url, json=None, timeout=None):
            if url.endswith('/motor/heartbeat'):
                state['beats'] += 1
                if state['beats'] == 3:
                    pi.command = 'stop'                       # Pi safety monitor fired
            return original(url, json, timeout)
        pi.post = flaky
        pi.get = lambda url, timeout=None: FakeResponse(200, {'obstacle_stop': False} if 'sensor' in url
                                                         else pi.status())
        problems = []
        legs_ctrl = controller(pi)
        legs_ctrl.drive(legs.make_intent('forward', seconds=1.5), problems.append)
        legs_ctrl._drive.join(4)
        self.assertEqual(problems, [])                        # nothing spoken
        self.assertEqual(pi.commands, ['forward', 'forward', 'stop'])   # resumed once

    def test_real_obstacle_is_spoken_with_distance(self):
        pi = FakePi(obstacle_after=0.2)
        pi.get = lambda url, timeout=None: FakeResponse(
            200, {'obstacle_stop': True, 'distance_1': 31.4, 'distance_2': None,
                  'obstacle_status': 'STOP - OBSTACLE TOO CLOSE'} if 'sensor' in url else pi.status())
        problems = []
        legs_ctrl = controller(pi)
        legs_ctrl.drive(legs.make_intent('forward', seconds=3), problems.append)
        legs_ctrl._drive.join(4)
        self.assertIn('obstacle ahead: 31 cm away', problems[0])

    def test_unreachable_pi_does_not_crash_execute(self):
        class DeadPi(object):
            def get(self, *a, **k):
                raise IOError('connection refused')
            post = get
        legs_ctrl = legs.LegsController('http://10.0.0.2:5000', http=DeadPi(), log=lambda *a: None)
        self.assertIn('connection refused', legs_ctrl.execute(legs.make_intent('forward')))

    def test_unreachable_pi_does_not_crash(self):
        class DeadPi(object):
            def get(self, *a, **k):
                raise IOError('connection refused')
            post = get
        legs_ctrl = legs.LegsController('http://10.0.0.2:5000', http=DeadPi(), log=lambda *a: None)
        with self.assertRaises(IOError):
            legs_ctrl.drive(legs.make_intent('forward'))
        legs_ctrl.stop()                                    # never raises

    def test_speed_and_unconfigured(self):
        pi = FakePi()
        self.assertIn('2.5 km/h', controller(pi).execute(legs.make_intent('faster')))
        logs = []
        self.assertIsNone(legs.connect_from_environment(log=logs.append, environ={}))
        self.assertIn('AISANA_LEGS_URL', logs[0])

    def test_replies(self):
        self.assertEqual(legs.legs_reply(legs.make_intent('forward'), 'Russian'), 'Еду вперёд.')
        self.assertEqual(legs.legs_reply(legs.make_intent('spin_left'), 'English'), 'Turning left.')
        self.assertEqual(legs.legs_reply(legs.parse_legs_command('развернись'), 'Russian'), 'Уииииии!')
        self.assertEqual(legs.legs_reply(legs.parse_legs_command('покрутись 10 секунд'), 'Russian'), 'Уииииии!')
        self.assertEqual(legs.legs_reply(legs.parse_legs_command('повернись налево'), 'Russian'),
                         'Поворачиваюсь налево.')



class SpinWordsTests(unittest.TestCase):
    def test_kruzhis(self):
        # Jetson log, Nov 23: "Айсана, покружись вокруг себя восемь секунд." did nothing.
        self.assertEqual(legs.parse_legs_command('Айсана, покружись вокруг себя восемь секунд.'),
                         legs.make_intent('spin_right', seconds=8.0))
        self.assertEqual(legs.parse_legs_command('Айсана, кружись')['degrees'], 360)
        self.assertIsNone(legs.parse_legs_command('Айсана, у меня голова кружится?'))

class StatePi(object):
    """The Pi's /motor API as odrive_controller.py answers it, state by state."""

    def __init__(self, state, enabled=False, error=None, connecting=False):
        self.state, self.enabled, self.error, self.connecting = state, enabled, error, connecting
        self.posts = []

    def status(self):
        return {'state': self.state, 'enabled': self.enabled, 'error': self.error,
                'connecting': self.connecting, 'command': 'stop', 'speed_kmh': 2.0}

    def get(self, url, timeout=None):
        return FakeResponse(200, self.status())

    def post(self, url, json=None, timeout=None):
        path = url.split(':5000', 1)[1]
        self.posts.append((path, json))
        return FakeResponse(200, self.status())


class KeepAliveTests(unittest.TestCase):
    """Nov 23: "the wheels only work while the Pi /control page is open"."""

    def controller(self, pi):
        c = legs.LegsController('http://pi:5000', http=pi, log=lambda text: None)
        c._closed.wait = lambda seconds: None          # no real sleeping in tests
        return c

    def test_keeps_enabled_motors_on_with_heartbeats(self):
        pi = StatePi('ENABLED', enabled=True)
        c = self.controller(pi)
        self.assertEqual(c.keepalive_step(now=0), legs.KEEPALIVE_SECONDS)
        c.keepalive_step(now=0.15)
        c.keepalive_step(now=0.30)
        self.assertEqual([p for p, _ in pi.posts], ['/motor/heartbeat', '/motor/heartbeat'])
        self.assertLess(legs.KEEPALIVE_SECONDS, 0.35)    # faster than the Pi's stop timeout

    def test_switches_motors_on_again_without_calibration(self):
        pi = StatePi('CONTROL LINK LOST', enabled=False)
        c = self.controller(pi)
        c.keepalive_step(now=0)
        self.assertEqual(pi.posts, [('/motor/enable', {'calibrate': False})])

    def test_waits_while_the_pi_connects(self):
        pi = StatePi('CONNECTING / ENABLING', connecting=True)
        c = self.controller(pi)
        c.keepalive_step(now=0)
        self.assertEqual(pi.posts, [])                    # no second enable or calibration on top

    def test_calibrates_by_itself_at_most_twice(self):
        pi = StatePi('ERROR', error='M0 (Left) calibration failed; axis.error=256')
        c = self.controller(pi)
        warned = []
        c.on_calibrate = lambda: warned.append(True)
        for t in (0, 10, 20, 30, 40):
            c.keepalive_step(now=t)
        self.assertEqual(pi.posts, [('/motor/enable', {'calibrate': True})] * 2)   # a broken encoder must not loop
        self.assertEqual(len(warned), 2)                   # Aisana warns before the wheels spin
        c.keepalive_step(now=10 + legs.AUTO_CALIBRATION_WINDOW + 1)
        self.assertEqual(len(pi.posts), 3)                 # tries again after 10 minutes

    def test_no_auto_calibration_when_switched_off(self):
        pi = StatePi('NEEDS CALIBRATION')
        c = self.controller(pi)
        c.auto_calibrate = False
        c.keepalive_step(now=0)
        self.assertEqual(pi.posts, [])



if __name__ == '__main__':
    unittest.main()
