"""Offline tests for realtime wake gating and tools; no network or audio."""
import json
import time
import unittest

import aisana_realtime as rt


class FakeArm(object):
    def __init__(self):
        self.calls = []
        self.stopped = 0

    def execute(self, action, arm='both', position=None, joint='whole',
                direction=None, degrees=None):
        self.calls.append((action, arm, position, joint, direction, degrees))
        return 'Sent to the Arduino.'

    def execute_many(self, intents, verify=True):
        self.calls.extend((i['action'], i['arm'], i['position'], i['joint'], i['direction'], i['degrees'])
                          for i in intents)
        return 'Sent to the Arduino. Feedback: right shoulder: 47 -> 61, target 77, still moving.'

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


def say(session, item_id, transcript):
    session.handle({'type': 'input_audio_buffer.committed', 'item_id': item_id})
    session.handle({'type': 'conversation.item.input_audio_transcription.completed',
                    'item_id': item_id, 'transcript': transcript})


class ArmVerificationTests(unittest.TestCase):
    """The case from the Jetson log of 2026-11-19: the model chose a nudge."""

    def test_users_exact_words_override_model_guess(self):
        session, _ = make_session()
        session.arm = FakeArm()
        say(session, 'i1', 'Айсана, подвигай плечом. Плечо правое в позицию 1. Позиция 4. '
                           'Айсана, правое плечо в позицию 2.')
        result = session.run_tool('move_arm', {'action': 'nudge', 'arm': 'right', 'joint': 'shoulder',
                                               'direction': 'up', 'degrees': 10})
        self.assertEqual([c[:4] for c in session.arm.calls],
                         [('position', 'right', 1, 'shoulder'), ('position', 'right', 4, 'shoulder'),
                          ('position', 'right', 2, 'shoulder')])
        self.assertIn('still moving', result)

    def test_arm_tool_refused_when_user_did_not_ask(self):
        session, _ = make_session()
        session.arm = FakeArm()
        say(session, 'i2', 'Айсана, какая сегодня погода?')
        result = session.run_tool('move_arm', {'action': 'position', 'arm': 'both', 'position': 3})
        self.assertEqual(session.arm.calls, [])
        self.assertIn('NOT moved', result)

    def test_vague_arm_request_uses_model_interpretation(self):
        session, _ = make_session()
        session.arm = FakeArm()
        say(session, 'i3', 'Айсана, помаши правой рукой')
        session.run_tool('move_arm', {'action': 'nudge', 'arm': 'right', 'direction': 'up', 'degrees': 15})
        self.assertEqual(session.arm.calls[0][:2], ('nudge', 'right'))

    def test_short_followup_more_allowed_after_recent_move(self):
        session, _ = make_session()
        session.arm = FakeArm()
        session.last_arm_at = time.monotonic()
        say(session, 'i4', 'ещё')
        session.run_tool('move_arm', {'action': 'nudge', 'arm': 'right', 'direction': 'up', 'degrees': 5})
        self.assertEqual(len(session.arm.calls), 1)

    def test_safety_net_runs_command_model_ignored(self):
        session, sent = make_session()
        session.arm = FakeArm()
        session.dialog_until = time.monotonic() + 20
        session.handle({'type': 'input_audio_buffer.committed', 'item_id': 'i5'})
        session.handle({'type': 'response.created'})
        session.handle({'type': 'response.done'})            # model answered with words only
        session.handle({'type': 'conversation.item.input_audio_transcription.completed',
                        'item_id': 'i5', 'transcript': 'левая рука в позицию 3'})
        time.sleep(0.2)
        self.assertEqual(session.arm.calls[0][:3], ('position', 'left', 3))
        self.assertIn('Robot report', sent[-2]['item']['content'][0]['text'])
        self.assertEqual(sent[-1]['type'], 'response.create')

    def test_no_double_move_when_model_called_tool(self):
        session, _ = make_session()
        session.arm = FakeArm()
        session.dialog_until = time.monotonic() + 20
        session.handle_tool_call = lambda *a: None
        session.handle({'type': 'input_audio_buffer.committed', 'item_id': 'i6'})
        session.handle({'type': 'response.created'})
        session.handle({'type': 'response.function_call_arguments.done', 'call_id': 'c9',
                        'name': 'move_arm', 'arguments': '{}'})
        session.handle({'type': 'response.done'})
        session.handle({'type': 'conversation.item.input_audio_transcription.completed',
                        'item_id': 'i6', 'transcript': 'левая рука в позицию 3'})
        time.sleep(0.2)
        self.assertEqual(session.arm.calls, [])            # tool thread owns it, net stays off

    def test_first_turn_with_arm_command_forces_tool(self):
        original = rt.WAKE_GATE
        rt.WAKE_GATE = True
        try:
            session, sent = make_session()
            say(session, 'i7', 'Айсана, правая рука в позицию 2')
            self.assertEqual(sent[-1]['response']['tool_choice']['name'], 'move_arm')
        finally:
            rt.WAKE_GATE = original


class ToolTests(unittest.TestCase):
    def test_tools_offered_to_model(self):
        names = [tool['name'] for tool in rt.tool_definitions()]
        for name in ('inspect_camera', 'move_arm', 'get_weather'):
            self.assertIn(name, names)
        arm_tool = [tool for tool in rt.tool_definitions() if tool['name'] == 'move_arm'][0]
        properties = arm_tool['parameters']['properties']
        self.assertEqual(properties['position']['maximum'], 4)
        self.assertIn('nudge', properties['action']['enum'])
        self.assertEqual(properties['degrees']['maximum'], 45)

    def test_move_arm_goes_through_controller(self):
        session, _ = make_session()
        session.arm = FakeArm()
        session.last_item = None            # no transcript: model arguments are used
        result = session.run_tool('move_arm', {'action': 'position', 'arm': 'left', 'position': 3})
        self.assertEqual(session.arm.calls, [('position', 'left', 3, 'whole', None, None)])
        session.run_tool('move_arm', {'action': 'nudge', 'arm': 'right', 'direction': 'up', 'degrees': 5})
        self.assertEqual(session.arm.calls[-1], ('nudge', 'right', None, 'whole', 'up', 5))
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
