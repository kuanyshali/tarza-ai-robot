# Sadam modernization: Jetson Nano validation

The wake name is required to **start** a conversation, but not for each follow-up during an active dialogue. The camera is not automatically narrated during ordinary chat.

## Update and validate

Stop the running robot with Ctrl+C, then:

```bash
cd ~/ai_robot
git pull origin main
python3 -m py_compile robot.py sadam_features.py
python3 -m unittest discover -s tests -v
python3 robot.py
```

Tests run offline without camera, audio hardware or an API key. Real-device checks below still need to be done on Jetson.

## Phases and test phrases

1. **Wake-name recovery**: 1.5 s audio pre-roll; primary multilingual `gpt-transcribe` transcription with name keywords. If the first transcription lacks the wake name, transcribe only the first 2.7 seconds of the recorded audio. Check the `Wake prefix transcription:` line. If neither contains the name, Sadam deliberately ignores the request.
2. **Multilingual conversation**: say `Садам, привет`, `Sadam, hello`, `صدام، إزيك؟`, and a Kazakh question. Test with a native Egyptian speaker.
3. **Visual questions**: say `Садам, что у меня в руке?`, `صدام، شايف إيه قدامك؟`, then `Садам, а в другой руке?`. Expected: `Camera routing: local YES` or `Camera classifier raw result: 'YES'`, then `Camera required: Yes`.
4. **Non-visual questions**: `Садам, какая погода в Астане?` should route to ordinary chat (not camera). The robot does not have live weather data unless a separate weather tool is added.
5. **Budget**: background cloud vision is now OFF by default. Camera capture and local change detection remain ON. To enable event descriptions explicitly, run `TARZA_BACKGROUND_ANALYSIS=1 python3 robot.py`.
6. **Image detail**: direct visual questions use a higher-resolution camera frame (default 960 px wide), instead of 320 px. Set `TARZA_VISION_IMAGE_WIDTH=640` to reduce bandwidth/cost, or `1280` for more detail.
7. **Timing**: logs include transcription, camera-routing and answer-generation times. Set `SADAM_DEBUG_TIMING=0` to disable.

## Configuration

- `TARZA_TRANSCRIBE_MODEL=gpt-transcribe` — recommended current speech-to-text model. To temporarily revert use `TARZA_TRANSCRIBE_MODEL=gpt-4o-mini-transcribe` (legacy, does not support language/keyword hints).
- `TARZA_PREROLL_SECONDS=1.5` — saves pre-trigger audio so softly spoken names are not clipped.
- `TARZA_WAKE_PREFIX_SECONDS=2.7` — short initial clip for wake-name recovery.
- `TARZA_SILENCE_SECONDS=1.15` — default waits for 1.15 seconds of silence before accepting a spoken segment. Set to `1.5` or `1.8` for speakers who pause frequently, or `0.9` for faster response.
- `TARZA_VOICE_RMS=350` — energy threshold for starting speech capture. Lower only if quiet wake words are cut off; background noise can cause false triggers.
- `TARZA_VISION_ENABLED=1` — camera worker on.
- `TARZA_BACKGROUND_ANALYSIS=0` — disable unsolicited cloud vision spending.
- `TARZA_VISION_IMAGE_WIDTH=960` — width of visual answer frames.
- `ROBOT_WAKE_WORDS` — optional comma-separated override; if set, it replaces the default wake variants.

## Xiaomi USB speaker: automatic identity, never KT

The confirmed Xiaomi currently appears in `aplay -l` as `USB Composite Device`. The robot finds its current ALSA card number on every playback and **never** falls back to `KT USB Audio` (the microphone adapter), HDMI or another generic USB output. If the exact Xiaomi label is absent or duplicated, voice playback fails with an explicit message while the listening loop continues. If a different device reports the same exact label, identification is not guaranteed; verify actual hardware before using it.

```bash
aplay -l
```

The microphone still uses `plughw:2,0`; that is independent of Xiaomi's playback card number.

## Continuous conversation and sentence endings

- Say **Садам / Саддам / Sadam / Saddam / صدام** to initiate a dialogue. Following replies may omit the wake name for **20 seconds after each spoken robot reply** (configurable via `TARZA_DIALOG_TIMEOUT=20`). After a timeout, the name is mandatory again.
- While the dialogue is open, the robot checks whether an unaddressed utterance is likely directed at it. The semantic intent check costs an additional small model call on ambiguous follow-ups and **is not speaker identification**; other people's direct questions or brief acknowledgements can still trigger the robot. For reliable speaker isolation, a dedicated voice verification system is needed.
- Say `отбой`, `спи`, or `go to sleep` to close the dialogue without stopping the robot process. `Ctrl+C` stops the application.
- Voice activation waits for **1.15 seconds of quiet** by default. After STT, if the sentence ends on a known incomplete conjunction (e.g. `потому что`, `because`, `если`, `and`), Sadam listens for another **2.5 seconds** for a continuation and combines the transcripts. This is a **heuristic**, not full semantic turn-end detection.
- `TARZA_CONTINUATION_WAIT=2.5`, `TARZA_MAX_CONTINUATIONS=2` and `TARZA_DIALOG_TIMEOUT=20` configure these behaviors. Longer waits trade reaction speed for less interruption.
- For strict wake-word-only behavior, set `TARZA_DIALOG_TIMEOUT=5` (minimum value), though the follow-up window is still nonzero.

## Latency optimization

- Obvious visual requests and simple greetings are routed **locally**, without another API call. Uncertain requests still use the semantic camera classifier.
- OpenAI TTS uses PCM HTTP streaming. Playback starts after an adjustable **1.8-second audio prebuffer**, instead of waiting for the whole TTS response. For network instability / ALSA underruns, increase the buffer, for example:
  ```bash
  SADAM_TTS_PREBUFFER_SECONDS=3.0 python3 robot.py
  ```
- Speech capture ends after 0.7 seconds of silence, adjustable with `TARZA_SILENCE_SECONDS`.
- Log `TTS first playback` is the relevant latency measurement. The streaming total also includes audio playback.
- Running `python3 -m unittest discover -s tests -v` is recommended before real-device tests. Jetson USB-audio behavior and network timing cannot be verified in offline tests.

## Live weather, online search and conversational voice interruption

This version adds `sadam_features.py` (Python 3.6 compatible).

- **Live weather** via Open-Meteo geocoding + current forecast. Ask `Садам, какая погода в Астане?`, `Sadam, weather in Dubai tomorrow?`, `Садам, ауа райы қандай?`, or `صدام، الجو عامل إيه؟`. With no city, the configured default is Astana. Defaults can be changed using `SADAM_DEFAULT_CITY`. Source is printed in the terminal; failed API requests produce an explicit retry response rather than fabricated measurements. Open-Meteo may require an appropriate license for commercial projects.
- **Live web answers** via OpenAI's documented `gpt-5-search-api` Chat Completions integration with `web_search_options`. News, elections, results, presidents, and other current facts trigger search. Its citations/links are printed to the console. Search adds API usage charges and latency; configure `SADAM_WEB_MODEL` for an available search-supported model or disable via `SADAM_WEB_ENABLED=0`. The previous `gpt-6.1-terra` remains the default for ordinary conversation and images.
- **Direct personality:** More natural, concise replies; challenge and controversial topics are answered factually rather than with boilerplate refusals. Current results require live data to avoid guessing.
- **Semantic turn ends:** At 1.15 seconds of silence, a small model assesses if a phrase is unfinished (except very short inputs). If it says WAIT, Sadam listens up to `TARZA_CONTINUATION_WAIT=2.5` seconds and joins the transcript, up to two times. This adds a small API call and a little latency; set `SADAM_SEMANTIC_END=0` to use only the existing inexpensive heuristic.
- **Barge-in:** While Xiaomi is audibly talking, the Jetson microphone listens in another thread. Saying **`Садам, стоп`** (or **`Sadam, stop`**, `صدام، اسكت`) may interrupt playback; saying `Садам, <new question>` interrupts and processes the new question. A wake word is mandatory to reduce accidental interruption from background speech. No claim of speaker/voice identity verification. This feature may be affected by acoustic feedback from the speaker; set `SADAM_BARGE_IN=0` to disable it if audio becomes unstable. Barge-in works while TTS is playing, not while the model is still generating a response.

### Validate on Jetson

```bash
cd ~/ai_robot
git pull origin main
python3 -m py_compile robot.py sadam_features.py
python3 -m unittest discover -s tests -v
python3 robot.py
```

Voice-check prompts: `Садам, какая погода в Астане?`, `Садам, кто победил вчера на выборах в Мичигане?`, `Садам, расскажи длинную историю...` followed mid-playback by `Садам, стоп`. Compare console lines `Tool: live weather`, `Tool: live web search`, `Voice interruption detected:` and `Audio output (Xiaomi)`.

If anything breaks, first try the safe fallback `SADAM_BARGE_IN=0 SADAM_SEMANTIC_END=0 python3 robot.py`, keeping camera, wake-name dialogue, search, and Xiaomi intact.

## GPT model settings

Default chat and vision: `gpt-6.1-terra`. Default camera router: `gpt-6-luna`. Change using environment variables `SADAM_CHAT_MODEL`, `SADAM_VISION_MODEL`, and `SADAM_ROUTER_MODEL`. Low-latency reasoning uses `SADAM_REASONING_EFFORT=none`. All API calls still require a working `OPENAI_API_KEY`.

## Known limitations

- Short-clip transcription cannot recover a wake name if the microphone/VAD never captured it. If the name still disappears, stop the robot after a failed attempt, play back `speech.wav` with `aplay -D plughw:<speaker-card>,0 speech.wav` (replace the number using `aplay -l`), and inspect the beginning of the recording. Only share voice samples with the speaker's permission.
- Cloud STT and semantic camera routing each add API latency and charges; the second short STT call happens only when the name is missing. Keyword hints are not proof that a word was spoken; confirm native-speaker accuracy on-device.
- Egyptian Arabic speech synthesis with the existing `tts-1/alloy` voice may retain an accent.
- Higher image detail increases visual API cost compared with low-detail frames.
- Live external data (weather, elections) needs separate tools; the model must not guess it.
