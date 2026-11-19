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
        self.verified = verify
        return 'Sent to the Arduino.'

    def watch(self, intents, report, first=0.8, second=2.5):
        self.watched = intents
        self.report = report

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
        # From the Jetson log: the name came at the end of the phrase.
        self.assertTrue(rt.has_wake_name('Позицию ноль верни. Айсана, позиция ноль.'))
        self.assertTrue(rt.has_wake_name('подними руки, Aisana'))
        for text in ('что ты видишь?', 'я сказал айсане привет', 'поговорим про айсану',
                     'hello there', ''):
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


class ReconnectTests(unittest.TestCase):
    def test_dropped_connection_reconnects_and_keeps_devices(self):
        runs, sleeps = [], []
        arm, legs = FakeArm(), FakeLegs()
        original = (rt.connect_from_environment, rt.aisana_legs.connect_from_environment,
                    rt.RealtimeAisana.run, rt.time.sleep, rt.KEY, rt.websocket)

        def fake_run(session, manage_devices=True):
            runs.append((session.arm, session.legs, manage_devices))
            if len(runs) == 1:
                raise IOError('The write operation timed out')
            raise KeyboardInterrupt

        try:
            rt.connect_from_environment = lambda: arm
            rt.aisana_legs.connect_from_environment = lambda: legs
            rt.RealtimeAisana.run = fake_run
            rt.time.sleep = sleeps.append
            rt.KEY, rt.websocket = 'key', object()
            with self.assertRaises(KeyboardInterrupt):
                rt.main()
        finally:
            (rt.connect_from_environment, rt.aisana_legs.connect_from_environment,
             rt.RealtimeAisana.run, rt.time.sleep, rt.KEY, rt.websocket) = original
        self.assertEqual(len(runs), 2)                      # reconnected once
        self.assertTrue(all(r == (arm, legs, False) for r in runs))   # same devices
        self.assertEqual(len(sleeps), 1)
        self.assertEqual(legs.stops, 1)                     # wheels stopped on exit


class LanguageAndSilenceTests(unittest.TestCase):
    def test_reply_language_follows_user_words(self):
        self.assertEqual(rt.language_of('Aisana, go forward for one second.'), 'English')
        self.assertEqual(rt.language_of('Айсана, проедь вперёд'), 'Russian')
        self.assertEqual(rt.language_of('Ана нәрсе автомат'), 'Kazakh')
        self.assertEqual(rt.language_of('آيسانا، روحي قدام'), 'Egyptian Arabic')
        session, sent = make_session()
        session.legs = FakeLegs()
        say(session, 'g1', 'Aisana, go forward for one second.')
        session.handle_tool_call('c1', 'move_legs', json.dumps({'action': 'forward', 'seconds': 1}))
        self.assertTrue(sent[-1]['response']['instructions'].startswith('Reply in English only'))
        result = sent[-2]['item']['output']
        self.assertIn('Driving forward.', result)          # English example, not only "Еду вперёд."

    def test_stay_silent_for_chatter(self):
        session, sent = make_session()
        say(session, 's1', 'Инерция же.')
        session.handle({'type': 'response.function_call_arguments.done', 'call_id': 'c2',
                        'name': 'stay_silent', 'arguments': '{}'})
        self.assertEqual(sent[-1]['item']['output'], 'silent')
        self.assertNotEqual(sent[-1]['type'], 'response.create')

    def test_cannot_stay_silent_when_called_by_name(self):
        session, sent = make_session()
        say(session, 's2', 'Айсана, привет')
        session.handle({'type': 'response.function_call_arguments.done', 'call_id': 'c3',
                        'name': 'stay_silent', 'arguments': '{}'})
        self.assertEqual(sent[-1]['type'], 'response.create')

    def test_tools_include_stay_silent(self):
        self.assertIn('stay_silent', [tool['name'] for tool in rt.tool_definitions()])


class VoiceStyleTests(unittest.TestCase):
    def test_voice_style_presets(self):
        import os
        old = os.environ.pop('AISANA_VOICE_STYLE', None)
        try:
            self.assertEqual(rt.voice_style_instructions(), '')
            os.environ['AISANA_VOICE_STYLE'] = 'cute'
            self.assertIn('anime', rt.voice_style_instructions())
            os.environ['AISANA_VOICE_STYLE'] = 'whisper a little'
            self.assertEqual(rt.voice_style_instructions(), 'VOICE STYLE: whisper a little ')
        finally:
            os.environ.pop('AISANA_VOICE_STYLE', None)
            if old is not None:
                os.environ['AISANA_VOICE_STYLE'] = old


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
        self.assertIn('right arm shoulder -> position 2', result)
        self.assertIn('few words', result)            # short spoken reply, no long report
        self.assertFalse(session.arm.verified)       # no 2 s sensor wait before speaking
        self.assertEqual(len(session.arm.watched), 3) # sensors checked in the background

    def test_model_may_not_invent_position(self):
        # From the Jetson log: "Рука правая." became "right arm -> position 2".
        session, _ = make_session()
        session.arm = FakeArm()
        say(session, 'i8', 'Рука правая.')
        result = session.run_tool('move_arm', {'action': 'position', 'arm': 'right', 'position': 2})
        self.assertEqual(session.arm.calls, [])
        self.assertIn('В какую позицию', result)

    def test_sensor_problem_is_spoken(self):
        session, sent = make_session()
        session.arm = FakeArm()
        say(session, 'i9', 'руки в позицию ноль')
        session.run_tool('move_arm', {'action': 'position', 'arm': 'both', 'position': 0})
        session.arm.report(['LEFT STEPPER STOPPED: moving the wrong way.'])
        time.sleep(1.0)
        self.assertIn('wrong way', sent[-2]['item']['content'][0]['text'])
        self.assertEqual(sent[-1]['type'], 'response.create')

    def test_arm_tool_refused_when_user_did_not_ask(self):
        session, _ = make_session()
        session.arm = FakeArm()
        say(session, 'i2', 'Айсана, какая сегодня погода?')
        result = session.run_tool('move_arm', {'action': 'position', 'arm': 'both', 'position': 3})
        self.assertEqual(session.arm.calls, [])
        self.assertIn('Nothing moved', result)

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


class FakeLegs(object):
    def __init__(self):
        self.intents = []
        self.stops = 0

    def execute(self, intent, report=None):
        self.intents.append(intent)
        return 'Driving {} for 2.0 s (stops by itself).'.format(intent['action'])

    def stop(self):
        self.stops += 1


class LegsToolTests(unittest.TestCase):
    def test_users_words_decide_the_drive(self):
        session, _ = make_session()
        session.legs = FakeLegs()
        say(session, 'l1', 'Айсана, проедь вперёд 3 секунды')
        result = session.run_tool('move_legs', {'action': 'forward', 'seconds': 8})
        self.assertEqual(session.legs.intents[0]['seconds'], 3.0)
        self.assertIn('few words', result)

    def test_model_cannot_drive_when_not_asked(self):
        session, _ = make_session()
        session.legs = FakeLegs()
        say(session, 'l2', 'Айсана, что у тебя впереди?')
        result = session.run_tool('move_legs', {'action': 'forward'})
        self.assertEqual(session.legs.intents, [])
        self.assertIn('Nothing moved', result)

    def test_model_may_not_invent_duration(self):
        session, _ = make_session()
        session.legs = FakeLegs()
        say(session, 'l3', 'Айсана, покатайся по комнате вперёд и обратно')
        session.run_tool('move_legs', {'action': 'forward', 'seconds': 10})
        self.assertIsNone(session.legs.intents[0]['seconds'])

    def test_stop_stops_wheels_and_arms(self):
        session, _ = make_session()
        session.legs, session.arm = FakeLegs(), FakeArm()
        say(session, 'l4', 'Айсана, стой!')
        self.assertEqual((session.legs.stops, session.arm.stopped), (1, 1))

    def test_first_turn_drive_forces_legs_tool(self):
        original = rt.WAKE_GATE
        rt.WAKE_GATE = True
        try:
            session, sent = make_session()
            say(session, 'l5', 'Айсана, повернись налево')
            self.assertEqual(sent[-1]['response']['tool_choice']['name'], 'move_legs')
        finally:
            rt.WAKE_GATE = original

    def test_empty_speech_reply_is_cancelled(self):
        # From the Jetson log: "You: " (noise) got "I'm here..." in English.
        session, sent = make_session()
        session.dialog_until = time.monotonic() + 20
        session.handle({'type': 'input_audio_buffer.committed', 'item_id': 'n1'})
        session.handle({'type': 'response.created'})
        session.handle({'type': 'conversation.item.input_audio_transcription.completed',
                        'item_id': 'n1', 'transcript': ''})
        self.assertIn({'type': 'response.cancel'}, sent)
        self.assertEqual(sent[-1], {'type': 'conversation.item.delete', 'item_id': 'n1'})

    def test_problem_report_waits_for_pending_tool_answer(self):
        session, sent = make_session()
        session.expect_tool_response = True            # tool answer requested, not created yet
        session.report_problem('Legs report: ', ['blocked'])
        time.sleep(0.3)
        self.assertEqual(sent, [])                     # did not collide with the tool answer
        session.expect_tool_response = False
        time.sleep(1.0)
        self.assertEqual(sent[-1]['type'], 'response.create')

    def test_legs_not_configured(self):
        session, _ = make_session()
        say(session, 'l6', 'Айсана, вперёд')
        self.assertIn('not connected', session.run_tool('move_legs', {'action': 'forward'}))


class ToolTests(unittest.TestCase):
    def test_tools_offered_to_model(self):
        names = [tool['name'] for tool in rt.tool_definitions()]
        for name in ('inspect_camera', 'move_arm', 'move_legs', 'get_weather'):
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
        self.assertIn('left arm -> position 3', result)

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
