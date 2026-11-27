# Aisana Realtime speech-to-speech prototype

This is the **fast mode**. Microphone PCM streams to OpenAI Realtime and generated PCM is played as it arrives. Tools: camera, the Arm team's Arduino arms over USB, Open-Meteo weather and web search. SQLite memory is not connected yet. See [README.md](README.md) for arms and settings.

A conversation starts with the name **Aisana / Айсана**; after each reply follow-ups need no name for `AISANA_DIALOG_TIMEOUT` (20) seconds. Speech that is not addressed to her is deleted from the conversation. `AISANA_REALTIME_WAKE=0` restores the old answer-everything behaviour. "Айсана, стоп" immediately sends the arms an emergency stop.

## Jetson Nano / Python 3.6

```bash
cd ~/ai_robot
git pull --ff-only origin main
.venv/bin/python -m pip install "pyserial==3.5" "websocket-client==1.3.1"
.venv/bin/python -u aisana_realtime.py 2>&1 | tee realtime.log
```

Requires an existing OPENAI_API_KEY environment variable, KT USB Audio mic, Xiaomi USB Composite Device speaker and ALSA arecord/aplay.

Default model: gpt-realtime-2.1 (full; gpt-realtime-2.1-mini is ~3x cheaper). Override with AISANA_REALTIME_MODEL. Default voice: marin. Override with AISANA_REALTIME_VOICE. Stop with Ctrl+C.

**Test with headphones or low speaker volume.** For experimental interruptions:

```bash
AISANA_REALTIME_BARGE_IN=1 .venv/bin/python -u aisana_realtime.py
```

Echo cancellation and accurate conversation.item.truncate are **not** implemented; the speaker can trigger false interruptions and the conversation state may not exactly match played audio. Keep barge-in disabled until tested with headphones. Do not leave the prototype unattended or connected indefinitely: continuous audio incurs API charges. 

Known limitations: early version; Realtime API responses and ALSA hardware have not been tested on the actual Jetson. If playback is cut short, capture the terminal logs and we will tune the playback drain. The legacy robot.py remains the fallback.
