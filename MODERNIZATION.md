# Sadam modernization: Jetson Nano validation

This release keeps **name required for every request**. The camera is not automatically narrated during ordinary chat.

## Update and validate

Stop the running robot with Ctrl+C, then:

```bash
cd ~/ai_robot
git pull origin main
python3 -m py_compile robot.py
python3 -m unittest discover -s tests -v
python3 robot.py
```

Tests run offline without camera, audio hardware or an API key. Real-device checks below still need to be done on Jetson.

## Phases and test phrases

1. **Wake-name recovery**: 1.5 s audio pre-roll; primary multilingual `gpt-transcribe` transcription with name keywords. If the first transcription lacks the wake name, transcribe only the first 2.7 seconds of the recorded audio. Check the `Wake prefix transcription:` line. If neither contains the name, Sadam deliberately ignores the request.
2. **Multilingual conversation**: say `Садам, привет`, `Sadam, hello`, `صدام، إزيك؟`, and a Kazakh question. Test with a native Egyptian speaker.
3. **Visual questions**: say `Садам, что у меня в руке?`, `صدام، شايف إيه قدامك؟`, then `Садам, а в другой руке?`. Expected: `Camera classifier raw result: 'YES'` and `Camera required: Yes`.
4. **Non-visual questions**: `Садам, какая погода в Астане?` should route to ordinary chat (not camera). The robot does not have live weather data unless a separate weather tool is added.
5. **Budget**: background cloud vision is now OFF by default. Camera capture and local change detection remain ON. To enable event descriptions explicitly, run `TARZA_BACKGROUND_ANALYSIS=1 python3 robot.py`.
6. **Image detail**: direct visual questions use a higher-resolution camera frame (default 960 px wide), instead of 320 px. Set `TARZA_VISION_IMAGE_WIDTH=640` to reduce bandwidth/cost, or `1280` for more detail.
7. **Timing**: logs include transcription, camera-routing and answer-generation times. Set `SADAM_DEBUG_TIMING=0` to disable.

## Configuration

- `TARZA_TRANSCRIBE_MODEL=gpt-transcribe` — recommended current speech-to-text model. To temporarily revert use `TARZA_TRANSCRIBE_MODEL=gpt-4o-mini-transcribe` (legacy, does not support language/keyword hints).
- `TARZA_PREROLL_SECONDS=1.5` — saves pre-trigger audio so softly spoken names are not clipped.
- `TARZA_WAKE_PREFIX_SECONDS=2.7` — short initial clip for wake-name recovery.
- `TARZA_SILENCE_SECONDS=1.0` — recording ends after a second of silence; raise to 1.5 if your friend speaks slowly or pauses mid-sentence.
- `TARZA_VOICE_RMS=350` — energy threshold for starting speech capture. Lower only if quiet wake words are cut off; background noise can cause false triggers.
- `TARZA_VISION_ENABLED=1` — camera worker on.
- `TARZA_BACKGROUND_ANALYSIS=0` — disable unsolicited cloud vision spending.
- `TARZA_VISION_IMAGE_WIDTH=960` — width of visual answer frames.
- `ROBOT_WAKE_WORDS` — optional comma-separated override; if set, it replaces the default wake variants.

## Known limitations

- Short-clip transcription cannot recover a wake name if the microphone/VAD never captured it. If the name still disappears, stop the robot after a failed attempt, play back `speech.wav` with `aplay -D plughw:3,0 speech.wav`, and inspect the beginning of the recording. Only share voice samples with the speaker's permission.
- Cloud STT and semantic camera routing each add API latency and charges; the second short STT call happens only when the name is missing. Keyword hints are not proof that a word was spoken; confirm native-speaker accuracy on-device.
- Egyptian Arabic speech synthesis with the existing `tts-1/alloy` voice may retain an accent.
- Higher image detail increases visual API cost compared with low-detail frames.
- Live external data (weather, elections) needs separate tools; the model must not guess it.
