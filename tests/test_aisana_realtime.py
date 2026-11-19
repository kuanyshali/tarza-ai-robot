"""Offline tests for realtime wake gating and tools; no network or audio."""
import json
import time
import unittest

import aisana_realtime as rt


class FakeArm(object):
    def __init__(self):
        self.calls = []
        self.stopped = 0

    def execute(self, action, arm='both', position=None, joint='whole'):
        self.calls.append((action, arm, position, joint))
        return 'Sent to the Arduino.'

    def emergency_stop(self):
        self.stopped += 1

    def close(self):
        pass


def make_session():
    session = rt.RealtimeAisana()
    sent = []
    session.send = sent.append
    return session, sent


def types(sent):
    return [message['type'] for message in sent]


class WakePhraseTests(unittest.TestCase):
    def test_wake_name_variants(self):
        for text in ('Айсана, что ты видишь?', 'эйсана привет', 'Aisana, hello',
                     'Окей, Айсана, какая погода?', 'آيسانا، إزيك؟'):
            with self.subTest(text=text):
                self.assertTrue(rt.has_wake_name(text))
        for text in ('что ты видишь?', 'я сказал айсане привет', 'hello there', ''):
            with self.subTest(text=text):
                self.assertFalse(rt.has_wake_name(text))

    def test_stt_prompt_echo_is_not_a_request(self):
        self.assertTrue(rt.is_prompt_echo('Айсана, Айсана, Айсана, Эйсана, آيسانا.'))
        self.assertFalse(rt.is_prompt_echo('Айсана, правая рука, позиция один.'))
        self.assertFalse(rt.is_prompt_echo('Айсана, привет'))

    def test_stop_and_sleep_only_for_short_phrases(self):
        self.assertTrue(rt.is_stop_phrase('Айсана, стоп!'))
        self.assertTrue(rt.is_stop_phrase('stop'))
        self.assertFalse(rt.is_stop_phrase('расскажи, почему автобус не сделал стоп у университета'))
        self.assertTrue(rt.is_sleep_phrase('Айсана, отбой'))


class WakeGateTests(unittest.TestCase):
    def setUp(self):
        self.original = rt.WAKE_GATE
        rt.WAKE_GATE = True

    def tearDown(self):
        rt.WAKE_GATE = self.original

    def test_unaddressed_speech_is_dropped(self):
        session, sent = make_session()
        session.handle({'type': 'input_audio_buffer.committed', 'item_id': 'i1'})
        self.assertEqual(sent, [])
        session.handle({'type': 'conversation.item.input_audio_transcription.completed',
                        'item_id': 'i1', 'transcript': 'ребята, пойдём обедать'})
        self.assertEqual(sent, [{'type': 'conversation.item.delete', 'item_id': 'i1'}])

    def test_name_starts_dialogue_and_followups_answer_immediately(self):
        session, sent = make_session()
        session.handle({'type': 'conversation.item.input_audio_transcription.completed',
                        'item_id': 'i1', 'transcript': 'Айсана, привет'})
        self.assertEqual(types(sent), ['response.create'])
        self.assertTrue(session.dialog_active())
        # Follow-up inside the window: respond at commit, not after STT.
        session.handle({'type': 'input_audio_buffer.committed', 'item_id': 'i2'})
        self.assertEqual(types(sent), ['response.create', 'response.create'])
        session.handle({'type': 'conversation.item.input_audio_transcription.completed',
                        'item_id': 'i2', 'transcript': 'а как дела?'})
        self.assertEqual(len(sent), 2)            # not answered twice

    def test_prompt_echo_dropped_even_with_names(self):
        session, sent = make_session()
        session.handle({'type': 'conversation.item.input_audio_transcription.completed',
                        'item_id': 'e1', 'transcript': 'Айсана, Айсана, Айсана, Эйсана, آيسانا.'})
        self.assertEqual(sent, [{'type': 'conversation.item.delete', 'item_id': 'e1'}])
        self.assertFalse(session.dialog_active())

    def test_window_expires(self):
        session, sent = make_session()
        session.dialog_until = time.monotonic() - 1
        session.handle({'type': 'input_audio_buffer.committed', 'item_id': 'i3'})
        self.assertEqual(sent, [])

    def test_voice_stop_freezes_arms(self):
        session, sent = make_session()
        session.arm = FakeArm()
        session.handle({'type': 'conversation.item.input_audio_transcription.completed',
                        'item_id': 'i4', 'transcript': 'Айсана, стоп'})
        self.assertEqual(session.arm.stopped, 1)


class ToolTests(unittest.TestCase):
    def test_tools_offered_to_model(self):
        names = [tool['name'] for tool in rt.tool_definitions()]
        for name in ('inspect_camera', 'move_arm', 'get_weather'):
            self.assertIn(name, names)
        arm_tool = [tool for tool in rt.tool_definitions() if tool['name'] == 'move_arm'][0]
        self.assertEqual(arm_tool['parameters']['properties']['position']['maximum'], 4)

    def test_move_arm_goes_through_controller(self):
        session, _ = make_session()
        session.arm = FakeArm()
        result = session.run_tool('move_arm', {'action': 'position', 'arm': 'left', 'position': 3})
        self.assertEqual(session.arm.calls, [('position', 'left', 3, 'whole')])
        self.assertIn('Arduino', result)

    def test_move_arm_without_usb_is_reported_not_faked(self):
        session, _ = make_session()
        self.assertIn('not connected', session.run_tool('move_arm', {'action': 'stop'}))

    def test_each_tool_call_runs_once(self):
        session, _ = make_session()
        ran = []
        session.handle_tool_call = lambda call_id, name, arguments: ran.append((call_id, name))
        args = json.dumps({'action': 'status'})
        session.handle({'type': 'response.function_call_arguments.done', 'call_id': 'c1',
                        'name': 'move_arm', 'arguments': args})
        session.handle({'type': 'response.output_item.done', 'item': {
            'type': 'function_call', 'call_id': 'c1', 'name': 'move_arm', 'arguments': args}})
        time.sleep(0.1)
        self.assertEqual(ran, [('c1', 'move_arm')])

    def test_tool_output_is_sent_back_and_answered(self):
        session, sent = make_session()
        session.arm = FakeArm()
        session.handle_tool_call('c2', 'move_arm', json.dumps({'action': 'gripper_open', 'arm': 'right'}))
        self.assertEqual(types(sent), ['conversation.item.create', 'response.create'])
        self.assertEqual(sent[0]['item']['call_id'], 'c2')

    def test_failed_tool_never_invents_result(self):
        session, sent = make_session()
        session.handle_tool_call('c3', 'unknown_tool', '{}')
        self.assertIn('failed', sent[0]['item']['output'])


if __name__ == '__main__':
    unittest.main()
