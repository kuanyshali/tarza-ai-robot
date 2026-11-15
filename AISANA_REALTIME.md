# Aisana Realtime speech-to-speech prototype

This is a **separate experimental entrypoint**. It does not replace robot.py, and does not yet connect camera, SQLite memory, weather, search or Arduino. It listens continuously and sends microphone PCM to OpenAI Realtime; generated PCM is played as it arrives.

## Jetson Nano / Python 3.6

```bash
cd ~/ai_robot
git pull --ff-only origin main
.venv/bin/python -m pip install "websocket-client==1.3.1"
.venv/bin/python aisana_realtime.py
```

Requires an existing OPENAI_API_KEY environment variable, KT USB Audio mic, Xiaomi USB Composite Device speaker and ALSA arecord/aplay.

Default model: gpt-realtime-2.1-mini. Override with AISANA_REALTIME_MODEL. Default voice: marin. Override with AISANA_REALTIME_VOICE. Stop with Ctrl+C.

**Test with headphones or low speaker volume.** For experimental interruptions:

```bash
AISANA_REALTIME_BARGE_IN=1 .venv/bin/python aisana_realtime.py
```

Echo cancellation and accurate conversation.item.truncate are **not** implemented; the speaker can trigger false interruptions and the conversation state may not exactly match played audio. Keep barge-in disabled until tested with headphones. The assistant currently responds to any speech (no wake-word gate). Do not leave the prototype unattended or connected indefinitely: continuous audio incurs API charges. Camera and physical actuator control remain unavailable in this mode.

Known limitations: early version; Realtime API responses and ALSA hardware have not been tested on the actual Jetson. If playback is cut short, capture the terminal logs and we will tune the playback drain. The legacy robot.py remains the fallback.
