# Project notes for Claude

Aisana (Айсана): university voice robot on an NVIDIA Jetson Nano. The owner talks in Russian.

## Hardware and runtime (do not break)
- Jetson Nano, JetPack 4.x, **Python 3.6** system interpreter. No 3.7+ syntax or stdlib
  (no `capture_output=`, `text=`, dataclasses, walrus, `asyncio.run`). CI runs `vermin -t=3.6-`.
- `cv2` comes from JetPack (GStreamer camera pipeline `nvarguscamerasrc`); never pip-install opencv.
  The camera is mounted upside down: `TARZA_CAMERA_ROTATE=180`.
- Microphone: ALSA label `KT USB Audio`. Speaker: Xiaomi, ALSA label `USB Composite Device`.
  Card numbers change after USB reconnects — always resolve by label, fail closed, never fall back.
- Arms: Arm team's Arduino Mega over USB serial, 115200 baud, firmware `robotic_arms_*.ino`
  (text commands: `b 1`, `l g o`, `r m 3`, `x` emergency stop, `p` status). Only whitelisted
  commands from `aisana_arm.build_command` may be sent. Opening the port resets the board;
  wait for `Both arms ready.`. Physical motion: keep "стоп" → `x` working in both modes.
- Repo lives at `~/ai_robot` on the Jetson; the owner updates with `git pull --ff-only origin main`.

## Code map
- `aisana_realtime.py` — main fast mode (OpenAI Realtime GA over websocket-client 1.3.1).
- `robot.py` — classic STT → chat → TTS loop; imports cv2 and needs OPENAI_API_KEY at import,
  so tests extract its functions via `ast` (see `tests/test_robot.py::isolated`).
- `aisana_arm.py`, `sadam_features.py`, `aisana_memory.py`, `wake_listener.py` are importable
  without hardware and tested directly.

## Workflow
- Run `python3 -m unittest discover -s tests -v` before every push; CI must be green.
- Work on a branch and open a pull request; do not push untested code to `main`.
- Hardware behaviour (audio, camera, serial, latency) cannot be tested in CI — say so and
  ask the owner for the Jetson console log after changes.
- Some OpenAI model names in the code (gpt-6.1-terra, gpt-6-luna, gpt-transcribe,
  gpt-realtime-2.1-mini) may be newer than your knowledge; do not "fix" them without checking docs.
