"""Offline tests for realtime wake gating and tools; no network or audio."""
import base64
import json
import time
import unittest

import aisana_realtime as rt

# Most tests were written for 'answer in the user's language'; English-only has its own tests.
rt.FIXED_LANGUAGE = None


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
    session.history = None              # never write the real ~/.local history in tests
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
                    rt.RealtimeAisana.run, rt.reconnect_pause, rt.KEY, rt.websocket)

        def fake_run(session, manage_devices=True):
            runs.append((session.arm, session.legs, manage_devices))
            if len(runs) == 1:
                raise IOError('The write operation timed out')
            raise KeyboardInterrupt

        try:
            rt.connect_from_environment = lambda: arm
            rt.aisana_legs.connect_from_environment = lambda: legs
            rt.RealtimeAisana.run = fake_run
            rt.reconnect_pause = sleeps.append
            rt.KEY, rt.websocket = 'key', object()
            with self.assertRaises(KeyboardInterrupt):
                rt.main()
        finally:
            (rt.connect_from_environment, rt.aisana_legs.connect_from_environment,
             rt.RealtimeAisana.run, rt.reconnect_pause, rt.KEY, rt.websocket) = original
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
        time.sleep(0.2)
        self.assertIn('"Driving forward."', sent[-1]['response']['instructions'])

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

    def test_shout_request_with_command(self):
        # Jetson log: "Вперёд 2 секунды и крикни «We»" only got "Еду вперёд."
        self.assertEqual(rt.shout_request('Вперёд 2 секунды и крикни «We».'), 'We')
        text = rt.confirm_instruction('Еду вперёд.', 'Вперёд 2 секунды и крикни «We».')
        self.assertIn('"Еду вперёд"', text)
        self.assertIn('"We!"', text)
        self.assertIsNone(rt.shout_request('Айсана, вперёд'))
        self.assertTrue(rt.has_wake_name('Асама, разведись.'))

    def test_tools_include_stay_silent(self):
        self.assertIn('stay_silent', [tool['name'] for tool in rt.tool_definitions()])


class FakeShouts(object):
    def __init__(self, pcm=b'\x01\x00' * 100):
        self.pcm = pcm
        self.played = []
        self.stopped = 0

    def play_for(self, seconds):
        if not self.pcm:
            return False
        self.played.append(seconds)
        return True

    def stop(self):
        self.stopped += 1


class MotionAndChatterTests(unittest.TestCase):
    """From the Jetson log of Nov 19, evening."""

    def setUp(self):
        self.original = rt.WAKE_GATE
        rt.WAKE_GATE = True

    def tearDown(self):
        rt.WAKE_GATE = self.original

    def test_spin_shouts_the_whole_time_and_says_nothing(self):
        session, sent = make_session()
        session.legs, session.shouts = FakeLegs(), FakeShouts()
        say(session, 'm1', 'Айсана, развернись')
        time.sleep(0.2)
        self.assertEqual(session.shouts.played, [2.0])        # FakeLegs drives 2.0 s
        self.assertNotIn('response.create', [m['type'] for m in sent])
        self.assertTrue(session.motion_active())

    def test_own_shouts_ignored_but_stop_works_while_moving(self):
        session, sent = make_session()
        session.legs, session.shouts, session.arm = FakeLegs(), FakeShouts(), FakeArm()
        session.motion_until = time.monotonic() + 5
        say(session, 'm2', 'Уииии вжух')
        self.assertEqual(session.legs.stops, 0)
        self.assertEqual(sent[-1], {'type': 'conversation.item.delete', 'item_id': 'm2'})
        say(session, 'm3', 'стоп')
        self.assertEqual(session.legs.stops, 1)
        self.assertGreaterEqual(session.shouts.stopped, 1)
        self.assertFalse(session.motion_active())

    def test_thinking_aloud_is_held_and_dropped_for_commands(self):
        # "Окей, дай мне мгновенье подумать." was played before "Уииии!".
        session, sent = make_session()
        session.legs, session.shouts = FakeLegs(), FakeShouts()
        session.dialog_until = time.monotonic() + 20
        session.handle({'type': 'input_audio_buffer.committed', 'item_id': 'h1'})
        session.handle({'type': 'response.created', 'response': {'id': 'r1'}})
        session.handle({'type': 'response.output_audio.delta', 'response_id': 'r1',
                        'delta': base64.b64encode(b'\x00\x01' * 50).decode()})
        self.assertTrue(session.audio_queue.empty())          # held, not played
        session.handle({'type': 'conversation.item.input_audio_transcription.completed',
                        'item_id': 'h1', 'transcript': 'развернись'})
        self.assertTrue(session.audio_queue.empty())          # dropped: it was a command

    def test_answer_to_a_question_is_released(self):
        session, sent = make_session()
        session.dialog_until = time.monotonic() + 20
        session.handle({'type': 'input_audio_buffer.committed', 'item_id': 'h2'})
        session.handle({'type': 'response.created', 'response': {'id': 'r2'}})
        session.handle({'type': 'response.output_audio.delta', 'response_id': 'r2',
                        'delta': base64.b64encode(b'\x00\x01' * 50).decode()})
        session.handle({'type': 'conversation.item.input_audio_transcription.completed',
                        'item_id': 'h2', 'transcript': 'а сколько тебе лет?'})
        self.assertFalse(session.audio_queue.empty())

    def test_shouts_recorded_silently_at_start(self):
        session, sent = make_session()
        session.handle({'type': 'session.updated'})
        request = sent[-1]['response']
        self.assertEqual((request['conversation'], request['metadata']['purpose']), ('none', 'shouts'))
        session.handle({'type': 'response.created', 'response': {'id': 'o1', 'metadata': {'purpose': 'shouts'}}})
        session.handle({'type': 'response.output_audio.delta', 'response_id': 'o1',
                        'delta': base64.b64encode(b'\x00\x01' * rt.RATE).decode()})
        self.assertTrue(session.audio_queue.empty())          # never played
        self.assertFalse(session.responding)
        session.handle({'type': 'response.done', 'response': {'id': 'o1'}})
        self.assertEqual(len(session.shouts.pcm), 2 * rt.RATE)

    def test_obstacle_phrase_is_exact(self):
        # "Я остановился из-за препятствия в двух метрах" for 29 cm.
        session, sent = make_session()
        session.user_language = 'Russian'
        session.report_legs_problem('obstacle ahead: 29 cm away')
        time.sleep(1.0)
        self.assertIn('Стоп! Препятствие в 29 сантиметрах.', sent[-1]['response']['instructions'])
        session.expect_tool_response = False          # that reply has been played
        session.report_legs_problem('obstacle ahead: 2 cm away')
        time.sleep(1.0)
        self.assertIn('Сработал датчик', sent[-1]['response']['instructions'])

    def test_repeat_last_drive_n_times(self):
        session, sent = make_session()
        session.legs, session.shouts = FakeLegs(), FakeShouts()
        session.last_drive = {'action': 'spin_right', 'seconds': None, 'meters': None, 'degrees': 180}
        session.last_drive_at = time.monotonic()
        say(session, 'p1', 'Айсана, повтори десять раз')
        time.sleep(0.2)
        self.assertEqual(session.legs.intents[0]['degrees'], 1800)

    def test_say_ka_is_not_a_shout(self):
        self.assertIsNone(rt.shout_request('Скажи-ка, покрутись 10 раз.'))
        self.assertEqual(rt.shout_request('крикни «Ура»'), 'Ура')


class HumorAndCameraTests(unittest.TestCase):
    def setUp(self):
        self.original = rt.WAKE_GATE
        rt.WAKE_GATE = True

    def tearDown(self):
        rt.WAKE_GATE = self.original

    def test_tars_style_humor_setting(self):
        text = rt.humor_instructions(75)
        self.assertIn('красавчик', text)
        self.assertIn('never joke about', text.replace("Never", 'never'))
        self.assertIn('NEVER mention the humor level', text)   # no "юмор 100%" in every answer
        self.assertIn('no jokes', rt.humor_instructions(10))

    def test_voice_command_changes_humor(self):
        session, sent = make_session()
        say(session, 'u1', 'Айсана, юмор 60 процентов')
        self.assertEqual(session.humor, 60)
        update = [m for m in sent if m['type'] == 'session.update'][0]
        self.assertIn('humor level is 60%', update['session']['instructions'])
        self.assertEqual(sent[-1]['type'], 'response.create')

    def test_who_is_in_front_answers_in_first_person_at_once(self):
        # Jetson log: "Секунду, я посмотрю..." then "На снимке изображён человек..."
        session, sent = make_session()
        session.inspect_camera = lambda question, detail=None: 'Вижу человека в тёмной футболке.'
        say(session, 'c1', 'Айсана, кто перед тобой?')
        time.sleep(0.2)
        self.assertIn('own eyes', sent[-2]['item']['content'][0]['text'])
        self.assertIn('FIRST PERSON', sent[-1]['response']['instructions'])
        self.assertIn('Never mention a photo', sent[-1]['response']['instructions'])
        # The model's own camera call for the same words stays silent (no double answer).
        self.assertTrue(session.run_tool('inspect_camera', {'question': 'кто?'}).startswith(rt.SILENT_RESULT))

    def test_late_model_reply_to_camera_question_is_cut(self):
        # Jetson log: the reply started at commit arrived after the transcript and
        # only "Сейчас быстро проверю..." was played; the real answer was lost.
        session, sent = make_session()
        session.inspect_camera = lambda question, detail=None: 'Вижу человека.'
        session.last_item = 'c2'
        say(session, 'c2', 'Айсана, кто перед тобой?')
        session.handle({'type': 'response.created', 'response': {'id': 'r1'}})
        self.assertIn({'type': 'response.cancel'}, sent)
        session.handle({'type': 'response.output_audio.delta', 'response_id': 'r1', 'delta': 'AAAA'})
        self.assertTrue(session.audio_queue.empty())
        session.handle({'type': 'response.done', 'response': {'id': 'r1'}})
        time.sleep(0.3)
        own = [m for m in sent if m['type'] == 'response.create' and 'FIRST PERSON' in str(m)]
        self.assertEqual(len(own), 1)
        # Our own answer is played, not cut.
        session.handle({'type': 'response.created', 'response': {'id': 'r2', 'metadata': rt.OWN}})
        self.assertFalse(session.playback_interrupted)

    def test_camera_phrases(self):
        for text in ('Айсана, кто перед тобой?', 'что ты сейчас видишь?', 'что у меня в руке?',
                     'прочитай что тут написано', 'what do you see?', 'who is in front of you?'):
            with self.subTest(text=text):
                self.assertTrue(rt.CAMERA_RE.search(text))
        self.assertFalse(rt.CAMERA_RE.search('как дела?'))
        self.assertTrue(rt.READ_RE.search('прочитай что тут написано'))
        self.assertFalse(rt.READ_RE.search('кто перед тобой?'))

    def test_her_own_shouts_are_ignored(self):
        # Jetson log: "Уи" -> "I'm staying quiet."
        for text in ('Уи', 'Уиииии!', 'Вжууух, уиии', 'Wheee!'):
            with self.subTest(text=text):
                self.assertTrue(rt.SHOUT_ECHO_RE.match(text))
        self.assertFalse(rt.SHOUT_ECHO_RE.match('Ура, получилось'))
        session, sent = make_session()
        session.dialog_until = time.monotonic() + 20
        say(session, 'w1', 'Уи')
        # The reply started at commit was held; it is dropped and never played.
        self.assertEqual(sent[-1], {'type': 'conversation.item.delete', 'item_id': 'w1'})
        self.assertIsNone(session.hold_item)
        self.assertTrue(session.audio_queue.empty())


class FakeStream(object):
    def __init__(self):
        self.frame = 'frame'

    def latest(self, max_age=1.5, wait=3.0):
        return self.frame


class FakeFaces(object):
    threshold = 1.0

    def __init__(self, name=None):
        self.name = name
        self.remembered = []
        self.forgotten = []
        self.things = []

    def faces(self, frame):
        return [{'box': (0, 0, 9, 9), 'vector': 'v', 'name': self.name, 'distance': 0.3, 'where': 'center'}]

    def identify(self, vector):
        return self.name, 0.3

    def remember(self, name, vectors):
        self.remembered.append((name, len(vectors)))
        self.name = name

    def learn(self, name, vector, distance):
        return False

    def forget(self, name):
        self.forgotten.append(name)
        self.name = None
        return 1

    def names(self):
        return [self.name] if self.name else []

    def things_text(self):
        return ''


class LogFixesTests(unittest.TestCase):
    """Jetson log of Nov 19, 23:40."""

    def setUp(self):
        self.original = rt.WAKE_GATE
        rt.WAKE_GATE = True

    def tearDown(self):
        rt.WAKE_GATE = self.original

    def test_own_answers_cannot_call_the_camera_again(self):
        # "Сейчас быстро посмотрю..." + inspect_camera instead of the real answer.
        session, sent = make_session()
        session.inspect_camera = lambda question, detail=None: 'Вижу человека.'
        say(session, 'c1', 'Айсана, кто перед тобой?')
        time.sleep(0.2)
        self.assertEqual(sent[-1]['response']['tool_choice'], 'none')

    def test_stray_word_in_dialogue_is_ignored(self):
        session, sent = make_session()
        session.dialog_until = time.monotonic() + 20
        say(session, 'p1', 'Pairing.')
        self.assertEqual(sent[-1], {'type': 'conversation.item.delete', 'item_id': 'p1'})
        for text in ('Вперёд', 'Назад', 'Развернись', 'Налево'):
            self.assertFalse(rt.is_stray_word(text), text)        # one-word commands still work
        for text in ('Да', 'спасибо', 'Почему?', 'Айсана'):
            self.assertFalse(rt.is_stray_word(text) and not rt.has_wake_name(text), text)

    def test_sana_at_the_start_of_a_sentence_is_her_name(self):
        # Jetson log (Nov 20, noisy room): "...конкретно. Сана, какая погода у нас?" was ignored.
        for text in ('Нет, у нас видоса-то вот этого нету конкретно. Сана, какая погода у нас?',
                     'Сана, развернись', 'Сана какая погода'):
            self.assertTrue(rt.has_wake_name(text), text)
        for text in ('Это моя подруга сана из группы', 'сана', 'Мы с Санами пойдём'):
            self.assertFalse(rt.has_wake_name(text), text)

    def test_stt_hint_echo_is_never_a_request(self):
        self.assertTrue(rt.is_prompt_echo(rt.STT_PROMPT))
        self.assertNotIn('развернись', rt.STT_PROMPT)            # no command can be echoed

    def test_misheard_turnaround(self):
        self.assertTrue(rt.has_wake_name('Айсам, развелись.'))
        for text in ('Айсана, развелись.', 'Айсана, разберись.'):
            self.assertEqual(rt.parse_legs_command(text)['action'], 'spin_right', text)
        self.assertIsNone(rt.parse_legs_command('Айсана, разберись с этим кодом'))


class HearingTests(unittest.TestCase):
    def test_new_stt_model_falls_back_when_refused(self):
        session, sent = make_session()
        session.stt_model = 'gpt-transcribe'
        session.handle({'type': 'error', 'error': {'message': "Invalid value: 'gpt-transcribe'. Supported transcription models: ..."}})
        self.assertEqual(session.stt_model, rt.STT_FALLBACK)
        self.assertEqual(sent[-1]['session']['audio']['input']['transcription']['model'], rt.STT_FALLBACK)
        sent[:] = []
        session.handle({'type': 'error', 'error': {'message': 'transcription failed again: model'}})
        self.assertEqual(sent, [])                       # only once

    def test_unrelated_errors_keep_the_model(self):
        session, sent = make_session()
        session.handle({'type': 'error', 'error': {'message': 'conversation_already_has_active_response'}})
        self.assertEqual(session.stt_model, rt.STT_MODEL)


class FakeHistory(object):
    def __init__(self, turns=()):
        self.turns = list(turns)
        self.saved = []
        self.cleared = False

    def recent_turns(self, limit=12):
        return self.turns[-limit:]

    def remember_turn(self, question, answer):
        self.saved.append((question, answer))

    def clear_history(self):
        self.cleared = True


class BackgroundTalkTests(unittest.TestCase):
    """Jetson log, Nov 20: Сегодня тачку пригнал с кореи. Айсана, как у тебя дела?"""

    def setUp(self):
        self.original = rt.WAKE_GATE
        rt.WAKE_GATE = True

    def tearDown(self):
        rt.WAKE_GATE = self.original

    def test_only_words_from_her_name_on(self):
        self.assertEqual(rt.request_part('Сегодня тачку пригнал с кореи. Айсана, как у тебя дела?'),
                         'Айсана, как у тебя дела?')
        for text in ('Скажи, Айсана, какая погода', 'Как у тебя дела, Айсана?', 'Окей, Айсана, вперёд',
                     'Ну короче. Айсана.', 'Это моя подруга Сана из группы. Пошли.'):
            self.assertEqual(rt.request_part(text), text)

    def test_she_answers_only_the_request(self):
        session, sent = make_session()
        say(session, 'b1', 'Сегодня тачку пригнал с кореи. Айсана, как у тебя дела?')
        deadline = time.time() + 3
        while not any(m.get('type') == 'response.create' for m in sent) and time.time() < deadline:
            time.sleep(0.02)
        self.assertIn({'type': 'conversation.item.delete', 'item_id': 'b1'}, sent)
        text = [m for m in sent if m['type'] == 'conversation.item.create'][-1]['item']['content'][0]['text']
        self.assertEqual(text, 'Айсана, как у тебя дела?')
        self.assertEqual(sent[-1]['response']['metadata'], rt.OWN)
        self.assertNotIn('tool_choice', sent[-1]['response'])     # weather/camera tools still allowed

    def test_command_after_background_talk_ignores_the_talk(self):
        session, sent = make_session()
        legs = FakeLegs()
        session.legs = legs
        say(session, 'b2', 'Вчера поворот проехали. Айсана, как дела?')
        time.sleep(0.2)
        self.assertEqual(legs.intents, [])                           # "поворот" before her name never drives


class EnglishAndGuestsTests(unittest.TestCase):
    """Nov 23: English only for the presentation; respectful greetings for titled guests."""

    def setUp(self):
        self.original = rt.WAKE_GATE, rt.FIXED_LANGUAGE
        rt.WAKE_GATE, rt.FIXED_LANGUAGE = True, 'English'

    def tearDown(self):
        rt.WAKE_GATE, rt.FIXED_LANGUAGE = self.original

    def test_english_even_for_russian_words(self):
        session, sent = make_session()
        self.assertEqual(session.user_language, 'English')
        say(session, 'e1', 'Айсана, как дела?')
        self.assertEqual(session.user_language, 'English')
        self.assertIn('always speak English only', session.instructions())
        self.assertIn('Reply in English only', session.reply_language())

    def test_guest_is_greeted_respectfully_and_sarcasm_is_off(self):
        session, sent = make_session()
        session.camera_stream, session.faces = FakeStream(), FakeFaces('Waqar Ahmad')
        session.faces.title = lambda name: 'President of Nazarbayev University' if name == 'Waqar Ahmad' else None
        session.ready = True
        original = rt.FACE_SCAN_SECONDS
        rt.FACE_SCAN_SECONDS = 0.05
        try:
            __import__('threading').Thread(target=session.watch_faces, daemon=True).start()
            deadline = time.time() + 3
            while not any('Waqar Ahmad, President of Nazarbayev University' in str(m) for m in sent) \
                    and time.time() < deadline:
                time.sleep(0.02)
            session.stop.set()
        finally:
            rt.FACE_SCAN_SECONDS = original
        greeting = [m for m in sent if 'President of Nazarbayev University' in str(m) and m['type'] == 'response.create']
        self.assertTrue(greeting)
        self.assertIn('Welcome', str(greeting[-1]))
        self.assertNotIn('no sarcasm', session.instructions())     # humor stays on (default)
        original_vip = rt.VIP_MINUTES
        rt.VIP_MINUTES = 15                                         # optional respectful mode
        try:
            session.vip_seen('Waqar Ahmad')
            self.assertIn('no sarcasm', session.instructions())
        finally:
            rt.VIP_MINUTES = original_vip


class StopAndGreetTests(unittest.TestCase):
    """Jetson log, Nov 23."""

    def setUp(self):
        self.original = rt.WAKE_GATE
        rt.WAKE_GATE = True

    def tearDown(self):
        rt.WAKE_GATE = self.original

    def test_stop_is_sent_once_and_confirmed_with_a_fixed_phrase(self):
        # "Асана, стоп" -> two EMERGENCY STOPs and "Опускаю руки в ноль."
        session, sent = make_session()
        session.arm = FakeArm()
        say(session, 's1', 'Асана, стоп.')
        self.assertEqual(session.arm.stopped, 1)
        self.assertIn('s1', session.local_items)
        result = session.run_tool('move_arm', {'action': 'stop', 'arm': 'right'})
        self.assertIn('Already executed', result)
        self.assertEqual(session.arm.stopped, 1)
        deadline = time.time() + 3
        while not any('Стою' in str(m) for m in sent) and time.time() < deadline:
            time.sleep(0.02)
        self.assertTrue(any('Стою' in str(m) for m in sent))

    def test_greeting_waves_the_hand(self):
        session, sent = make_session()
        session.arm = FakeArm()
        say(session, 'g1', 'Айсана, поздоровайся.')
        deadline = time.time() + 2
        while not session.arm.calls and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(session.arm.calls[0][0], 'hand_shake')
        session.stop.set()                              # ends the wave timer early

    def test_wave_ends_by_itself_and_lowers_the_arm(self):
        session, sent = make_session()
        session.arm = FakeArm()
        say(session, 'w1', 'Айсана, помаши рукой 1 секунду')
        deadline = time.time() + 4
        while ('position', 'right', 0, 'whole', None, None) not in session.arm.calls and time.time() < deadline:
            time.sleep(0.05)
        actions = [c[0] for c in session.arm.calls]
        self.assertEqual(actions, ['hand_shake', 'hand_still', 'position'])

    def test_a_new_arm_command_cancels_the_wave_timer(self):
        session, sent = make_session()
        session.arm = FakeArm()
        say(session, 'w2', 'Айсана, помаши рукой 1 секунду')
        time.sleep(0.3)
        say(session, 'w3', 'Айсана, левая рука позиция два')
        time.sleep(1.5)
        self.assertNotIn('hand_still', [c[0] for c in session.arm.calls])
        self.assertEqual(rt.wave_seconds('помаши рукой'), rt.WAVE_SECONDS)

    def test_dance_spins_with_arms_up_then_arms_down(self):
        session, sent = make_session()
        session.arm, session.legs = FakeArm(), FakeLegs()
        session.shouts = FakeShouts()
        say(session, 'd1', 'Айсана, потанцуй 1 секунду и подними руки')
        deadline = time.time() + 4
        while ('position', 'both', 0, 'whole', None, None) not in session.arm.calls and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(session.legs.intents[0]['action'], 'spin_right')
        self.assertEqual(session.legs.intents[0]['seconds'], 1.0)
        actions = [c[:4] for c in session.arm.calls]
        self.assertEqual(actions[0], ('position', 'both', 3, 'shoulder'))
        self.assertEqual(actions[1][0], 'hand_shake')
        self.assertEqual(actions[-1], ('position', 'both', 0, 'whole'))
        self.assertFalse(rt.DANCE_RE.search('Ты умеешь танцевать?'))

    def test_short_legs_errors(self):
        self.assertIn('no answer', rt.aisana_legs.short_error(Exception('Connection to 10.42.0.148 timed out.')))


class ArmReconnectTests(unittest.TestCase):
    def setUp(self):
        self.original = rt.WAKE_GATE
        rt.WAKE_GATE = True

    def tearDown(self):
        rt.WAKE_GATE = self.original

    def test_arm_link_is_retried_when_a_command_comes(self):
        # Jetson log, Nov 20: "NOT done: the arm controller is not connected over USB."
        session, sent = make_session()
        session.arm = None
        arm = FakeArm()
        kept = []
        session.on_arm_connected = kept.append
        original = rt.connect_from_environment
        rt.connect_from_environment = lambda: arm
        try:
            say(session, 'a1', 'Айсана, правую руку, позиция один.')
            deadline = time.time() + 3
            while not arm.calls and time.time() < deadline:
                time.sleep(0.02)
        finally:
            rt.connect_from_environment = original
        self.assertIs(session.arm, arm)
        self.assertEqual(kept, [arm])
        self.assertTrue(arm.calls)

    def test_harmless_cancel_error_is_not_printed(self):
        session, _ = make_session()
        import io, contextlib
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            session.handle({'type': 'error', 'error': {'code': 'response_cancel_not_active', 'message': 'x'}})
        self.assertEqual(out.getvalue(), '')


class LocalWakeTests(unittest.TestCase):
    def test_cloud_hears_only_after_the_name_or_inside_a_conversation(self):
        session, _ = make_session()
        self.assertTrue(session.streaming())                  # no spotter: as before
        session.spotter = object()
        self.assertFalse(session.streaming())
        session.stream_until = time.monotonic() + 5           # "Айсана" heard locally
        self.assertTrue(session.streaming())
        session.stream_until = 0
        session.dialog_until = time.monotonic() + 5           # she just answered
        self.assertTrue(session.streaming())
        session.dialog_until = 0
        session.motion_until = time.monotonic() + 5           # "стоп" must work while driving
        self.assertTrue(session.streaming())


class ConversationMemoryTests(unittest.TestCase):
    def setUp(self):
        self.original = rt.WAKE_GATE
        rt.WAKE_GATE = True

    def tearDown(self):
        rt.WAKE_GATE = self.original

    def test_earlier_conversations_are_loaded(self):
        session, sent = make_session()
        session.history = FakeHistory([(time.time() - 86400 * 1.5, 'Когда у нас пара?', 'В десять.')])
        session.load_history()
        text = sent[-1]['item']['content'][0]['text']
        self.assertIn('[yesterday] User: Когда у нас пара? | Aisana: В десять.', text)

    def test_answers_are_saved_but_not_noise_or_cut_replies(self):
        session, sent = make_session()
        session.history = FakeHistory()
        say(session, 'h1', 'Айсана, как дела?')
        session.handle({'type': 'response.created', 'response': {'id': 'r1'}})
        session.handle({'type': 'response.output_audio_transcript.done', 'response_id': 'r1', 'transcript': 'Отлично.'})
        self.assertEqual(session.history.saved, [('Айсана, как дела?', 'Отлично.')])
        session.handle({'type': 'response.output_audio_transcript.done', 'response_id': 'r1', 'transcript': 'Ещё раз.'})
        self.assertEqual(len(session.history.saved), 1)               # one answer per question
        session.cut_ids.add('r2')
        say(session, 'h2', 'Айсана, а погода?')
        session.handle({'type': 'response.output_audio_transcript.done', 'response_id': 'r2', 'transcript': 'Секунду'})
        self.assertEqual(len(session.history.saved), 1)

    def test_voice_command_clears_history(self):
        session, sent = make_session()
        session.history = FakeHistory()
        say(session, 'h3', 'Айсана, забудь наши разговоры')
        self.assertTrue(session.history.cleared)


class FaceMemoryRealtimeTests(unittest.TestCase):
    def setUp(self):
        self.original = rt.WAKE_GATE
        rt.WAKE_GATE = True

    def tearDown(self):
        rt.WAKE_GATE = self.original

    def session(self, name=None):
        session, sent = make_session()
        session.camera_stream, session.faces = FakeStream(), FakeFaces(name)
        session.face_vectors = lambda seconds=2.0: ['v1', 'v2', 'v3', 'v4']
        return session, sent

    def wait_for(self, sent, text):
        deadline = time.time() + 3
        while time.time() < deadline:
            if any(text in str(m) for m in sent):
                return
            time.sleep(0.02)
        self.fail('never sent: ' + text)

    def test_remember_me_only_on_request(self):
        session, sent = self.session()
        say(session, 'f1', 'Айсана, запомни меня, я Али')
        self.wait_for(sent, 'the face of Али is now remembered')
        self.assertEqual(session.faces.remembered, [('Али', 4)])
        self.assertIn('f1', session.local_items)

    def test_name_asked_when_missing(self):
        session, sent = self.session()
        say(session, 'f2', 'Айсана, запомни меня')
        self.wait_for(sent, 'the name is needed')
        session.dialog_until = time.monotonic() + 10
        say(session, 'f3', 'Меня зовут Айгерим')
        self.wait_for(sent, 'the face of Айгерим is now remembered')

    def test_forget_me(self):
        session, sent = self.session('Али')
        say(session, 'f4', 'Айсана, забудь меня')
        self.wait_for(sent, 'the face of Али was deleted')
        self.assertEqual(session.faces.forgotten, ['Али'])

    def test_camera_answer_uses_known_name(self):
        # Jetson log: "Faces: Коныш Али 0.26" but the answer was "я не знаю, кто это".
        session, sent = self.session('Али')
        session.inspect_camera = lambda question, detail=None: (session.people_note('frame'), 'Вижу человека.')[1]
        say(session, 'k1', 'Айсана, кто перед тобой?')
        self.wait_for(sent, 'FIRST PERSON')
        report = [m for m in sent if m['type'] == 'conversation.item.create'][-1]['item']['content'][0]['text']
        self.assertIn('this is Али (center)', report)
        self.assertNotIn('Али', session.people_note('frame'))      # the vision model never gets names
        session.faces.name = None
        session.people_note('frame')
        self.assertIn('never invent names', session.people_report())

    def test_camera_looks_while_the_user_speaks(self):
        session, sent = self.session('Али')
        looks = []
        session.inspect_camera = lambda question, detail=None, quiet=False: looks.append(question) or 'Вижу человека.'
        session.handle({'type': 'input_audio_buffer.speech_started'})
        time.sleep(0.1)
        self.assertEqual(looks, [])                          # strangers talking: no paid look
        session.dialog_until = time.monotonic() + 10
        session.prefetch_at = 0
        session.handle({'type': 'input_audio_buffer.speech_started'})
        time.sleep(0.1)
        self.assertEqual(len(looks), 1)                      # started before the words arrived
        say(session, 'pf1', 'Айсана, кто перед тобой?')
        self.wait_for(sent, 'FIRST PERSON')
        self.assertEqual(len(looks), 1)                      # no second look
        session.prefetch_at = 0
        session.dialog_until = time.monotonic() + 10
        session.handle({'type': 'input_audio_buffer.speech_started'})
        time.sleep(0.1)
        say(session, 'pf2', 'Айсана, прочитай что тут написано')
        time.sleep(0.3)
        self.assertEqual(len(looks), 3)                      # reading gets its own sharp look

    def test_memory_replies_are_fixed_phrases(self):
        session, sent = self.session()
        say(session, 'f9', 'Айсана, запомни меня, я Али')
        self.wait_for(sent, 'Say exactly this')
        text = sent[-1]['response']['instructions']
        self.assertIn('Али', text)
        self.assertNotIn('запомнил ', text)                         # feminine only

    def test_greets_a_known_face_once(self):
        session, sent = self.session('Али')
        session.ready = True
        original = rt.FACE_SCAN_SECONDS
        rt.FACE_SCAN_SECONDS = 0.05
        try:
            worker = __import__('threading').Thread(target=session.watch_faces, daemon=True)
            worker.start()
            self.wait_for(sent, 'Али (you know them) just came into view')
            time.sleep(0.3)
            session.stop.set()
        finally:
            rt.FACE_SCAN_SECONDS = original
        self.assertEqual(sum('just came into view' in str(m) for m in sent), 1)


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
        time.sleep(0.2)                               # executed at once from the words
        result = session.run_tool('move_arm', {'action': 'nudge', 'arm': 'right', 'joint': 'shoulder',
                                               'direction': 'up', 'degrees': 10})
        self.assertEqual([c[:4] for c in session.arm.calls],
                         [('position', 'right', 1, 'shoulder'), ('position', 'right', 4, 'shoulder'),
                          ('position', 'right', 2, 'shoulder')])
        self.assertIn('Already executed', result)     # the model's nudge guess is not run
        self.assertFalse(session.arm.verified)       # no 2 s sensor wait before speaking
        self.assertEqual(len(session.arm.watched), 3) # sensors checked in the background

    def test_model_tool_path_still_checks_words(self):
        # If the model calls move_arm before the transcript arrives, its guess is replaced.
        session, _ = make_session()
        session.arm = FakeArm()
        session.handle({'type': 'input_audio_buffer.committed', 'item_id': 'm1'})
        session.handle({'type': 'response.function_call_arguments.done', 'call_id': 'cm', 'name': 'move_arm',
                        'arguments': '{}'})
        session.handle_tool_call = lambda *a: None
        with session.transcript_cond:
            session.transcripts['m1'] = 'правое плечо в позицию 2'
        result = session.run_tool('move_arm', {'action': 'nudge', 'arm': 'right', 'direction': 'up', 'degrees': 10})
        self.assertEqual(session.arm.calls[0][:4], ('position', 'right', 2, 'shoulder'))
        self.assertIn('right arm shoulder -> position 2', result)

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
        say(session, 'i3', 'Айсана, подвигай правой рукой')
        session.run_tool('move_arm', {'action': 'nudge', 'arm': 'right', 'direction': 'up', 'degrees': 15})
        self.assertEqual(session.arm.calls[0][:2], ('nudge', 'right'))

    def test_short_followup_more_allowed_after_recent_move(self):
        session, _ = make_session()
        session.arm = FakeArm()
        session.last_arm_at = time.monotonic()
        say(session, 'i4', 'ещё')
        session.run_tool('move_arm', {'action': 'nudge', 'arm': 'right', 'direction': 'up', 'degrees': 5})
        self.assertEqual(len(session.arm.calls), 1)

    def test_clear_command_runs_even_if_model_ignored_it(self):
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

    def test_first_turn_arm_command_runs_without_model_chatter(self):
        original = rt.WAKE_GATE
        rt.WAKE_GATE = True
        try:
            session, sent = make_session()
            session.arm = FakeArm()
            say(session, 'i7', 'Айсана, правая рука в позицию 2')
            time.sleep(0.2)
            self.assertEqual(session.arm.calls[0][:3], ('position', 'right', 2))
            self.assertIn('Robot report', sent[-2]['item']['content'][0]['text'])
            self.assertEqual(sent[-1]['response'].get('tool_choice'), 'none')   # no second arm move
        finally:
            rt.WAKE_GATE = original

    def test_stt_hint_echo_never_moves_anything(self):
        # From the Jetson log: noise came back as the STT command list and opened the grippers.
        session, sent = make_session()
        session.arm, session.legs = FakeArm(), FakeLegs()
        session.dialog_until = time.monotonic() + 20
        say(session, 'e9', 'поверни налево, поверни направо, развернись, развернись на 360, проедь вперёд, '
                           'езжай назад, стой, руки в позицию, открой захват, turn left, go forward.')
        time.sleep(0.2)
        self.assertEqual((session.arm.calls, session.legs.intents), ([], []))


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
        time.sleep(0.2)
        result = session.run_tool('move_legs', {'action': 'forward', 'seconds': 8})
        self.assertEqual(session.legs.intents, [session.legs.intents[0]])   # once, from the words
        self.assertEqual(session.legs.intents[0]['seconds'], 3.0)
        self.assertIn('Already executed', result)

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
        session.handle({'type': 'input_audio_buffer.committed', 'item_id': 'l3'})
        with session.transcript_cond:
            session.transcripts['l3'] = 'Айсана, покатайся по комнате вперёд и обратно'
        session.run_tool('move_legs', {'action': 'forward', 'seconds': 10})
        self.assertIsNone(session.legs.intents[0]['seconds'])

    def test_stop_stops_wheels_and_arms(self):
        session, _ = make_session()
        session.legs, session.arm = FakeLegs(), FakeArm()
        say(session, 'l4', 'Айсана, стой!')
        self.assertEqual((session.legs.stops, session.arm.stopped), (1, 1))

    def test_spin_and_shout(self):
        original = rt.WAKE_GATE
        rt.WAKE_GATE = True
        try:
            session, sent = make_session()
            session.legs = FakeLegs()
            say(session, 'l5', 'Айсана, покрутись 10 секунд и кричи уииии')
            time.sleep(0.2)
            self.assertEqual(session.legs.intents[0]['action'], 'spin_right')
            self.assertEqual(session.legs.intents[0]['seconds'], 10.0)
            self.assertIn('Уииии', sent[-1]['response']['instructions'])
            # Short fixed phrases for other commands; "Уииииии!" for every turnaround.
            say(session, 'l7', 'Айсана, вперёд')
            time.sleep(0.2)
            self.assertIn('"Еду вперёд."', sent[-1]['response']['instructions'])
            self.assertIn('nothing else', sent[-1]['response']['instructions'])
            say(session, 'l8', 'Айсана, развернись')
            time.sleep(0.2)
            self.assertIn('"Уииииии!"', sent[-1]['response']['instructions'])
            say(session, 'l9', 'Aisana, turn around')
            time.sleep(0.2)
            self.assertIn('"Wheeeee!"', sent[-1]['response']['instructions'])
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
        session, sent = make_session()
        say(session, 'l6', 'Айсана, вперёд')
        time.sleep(0.2)
        self.assertIn('NOT done: legs are not connected', sent[-2]['item']['content'][0]['text'])


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
