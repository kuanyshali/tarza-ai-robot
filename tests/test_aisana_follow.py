"""'Иди за мной': decisions and the follow loop with a fake camera, detector and Pi."""
import threading
import time
import unittest

import aisana_follow as follow
import aisana_legs as legs


def box(cx, width, top=0.1, bottom=0.95):
    return (cx - width / 2.0, top, cx + width / 2.0, bottom, 0.9)


class DecideTests(unittest.TestCase):
    def test_turns_toward_the_person(self):
        command, pulse, _ = follow.decide(box(0.1, 0.2), walking=False)
        self.assertEqual(command, 'spin_left')               # person on her left
        self.assertTrue(0.08 <= pulse <= 0.35)
        self.assertEqual(follow.decide(box(0.9, 0.2), False)[0], 'spin_right')

    def test_walks_when_far_and_stops_when_close(self):
        self.assertEqual(follow.decide(box(0.5, 0.15), False)[0], 'forward')
        self.assertEqual(follow.decide(box(0.6, 0.15), False)[0], 'arc_right')
        self.assertEqual(follow.decide(box(0.5, 0.40), True)[0], 'stop')
        # hysteresis: just under NEAR it keeps walking, but from standing it waits
        width = (follow.NEAR_WIDTH + follow.FAR_WIDTH) / 2
        self.assertEqual(follow.decide(box(0.5, width), True)[0], 'forward')
        self.assertEqual(follow.decide(box(0.5, width), False)[0], 'stop')

    def test_keeps_the_same_person(self):
        me, other = box(0.3, 0.2), box(0.8, 0.35)
        self.assertEqual(follow.choose_target([me, other], None), other)   # biggest at start
        self.assertEqual(follow.choose_target([box(0.33, 0.2), other], me)[0], box(0.33, 0.2)[0])
        self.assertIsNone(follow.choose_target([], me))


class FakeLegs(object):
    spin_90 = 0.84

    def __init__(self, refuse=None):
        self.commands = []
        self.refuse = refuse or set()

    def ensure_enabled(self):
        return {'enabled': True}

    def set_command(self, command, lease=0.9):
        self.commands.append(command)
        if command in self.refuse:
            return 409, {'error': 'Movement blocked: obstacle closer than 40 cm.'}
        return 200, {'command': command}


class FakeDetector(object):
    def __init__(self, script):
        self.script = list(script)

    def detect(self, frame):
        return self.script.pop(0) if self.script else []


class FollowerTests(unittest.TestCase):
    def run_follower(self, script, refuse=None, seconds=5.0, lost=0.2):
        fake_legs, ended = FakeLegs(refuse), []
        follower = follow.Follower(fake_legs, lambda: 'frame', FakeDetector(script), log=lambda t: None,
                                   on_end=ended.append, seconds=seconds, lost_seconds=lost,
                                   sleep=lambda s: time.sleep(min(s, 0.01)))
        follower.run()
        return fake_legs.commands, ended

    def test_follows_then_stops_when_the_person_is_lost(self):
        commands, ended = self.run_follower([[box(0.5, 0.15)], [box(0.1, 0.15)], [box(0.5, 0.45)]])
        self.assertEqual(commands[0], 'forward')
        self.assertIn('spin_left', commands)
        self.assertEqual(commands[-1], 'stop')               # always ends standing
        self.assertEqual(ended, ['lost'])

    def test_waits_at_an_obstacle(self):
        commands, ended = self.run_follower([[box(0.5, 0.15)]] * 3, refuse={'forward'})
        self.assertEqual(commands.count('forward'), 3)       # refused (409): no crash, keeps trying
        self.assertEqual(commands[-1], 'stop')

    def test_cancel_stops_at_once(self):
        fake_legs = FakeLegs()
        detector = FakeDetector([[box(0.5, 0.15)]] * 10000)
        follower = follow.Follower(fake_legs, lambda: 'frame', detector, log=lambda t: None,
                                   seconds=30, sleep=lambda s: time.sleep(0.005)).start()
        time.sleep(0.05)
        follower.cancel()
        follower.thread.join(2)
        self.assertFalse(follower.active())
        self.assertEqual(fake_legs.commands[-1], 'stop')
        self.assertEqual(follower.reason, 'stopped')

    def test_time_limit(self):
        commands, ended = self.run_follower([[box(0.5, 0.15)]] * 100000, seconds=0.05)
        self.assertEqual(ended, ['time'])
        self.assertEqual(commands[-1], 'stop')


class LeaseTests(unittest.TestCase):
    """The keep-alive must never keep a drive going that nobody runs."""

    def test_unowned_drive_is_stopped_by_the_keepalive(self):
        posts = []

        class Pi(object):
            def get(self, url, timeout=None):
                raise AssertionError('not used')

            def post(self, url, json=None, timeout=None):
                posts.append((url.split(':5000', 1)[1], json))

                class R(object):
                    status_code = 200

                    def json(self):
                        return {'enabled': True, 'command': 'forward', 'state': 'ENABLED'}
                return R()
        c = legs.LegsController('http://pi:5000', http=Pi(), log=lambda t: None)
        c.alive_on = True
        c.motion_lease_until = time.monotonic() + 5          # a follow loop owns it
        c.keepalive_step()
        self.assertEqual(posts, [('/motor/heartbeat', {})])
        c.motion_lease_until = time.monotonic() - 0.1        # the loop died
        c.keepalive_step()
        self.assertIn(('/motor/command', {'command': 'stop'}), posts)


if __name__ == '__main__':
    unittest.main()
