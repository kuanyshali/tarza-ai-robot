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
  Latest firmware: robotic_arms_checked5.ino (Nov 23): RIGHT elbow has NO pot (positions = servo angles
  80,106,132,158,184; rest 80 = down; status 'servo N deg (no pot)', no feedback), RIGHT arm has a shaking
  HAND ('r g s' shake / 'r g n' stop) instead of a gripper; only the LEFT gripper opens/closes.
  Arm moves are verified with "p" before/after; never let the model claim motion without feedback.
  In realtime mode explicit commands are re-parsed from the STT transcript (aisana_arm.parse_arm_commands);
  model arguments are used only for vague arm requests.
- Legs: Raspberry Pi over LAN runs the legs team's Flask app (app.py + odrive_controller.py,
  port 5000, NOT in this repo). aisana_legs.py uses its HTTP API only (/motor/enable without
  calibration, /motor/command, /motor/heartbeat every 0.12 s; the Pi stops after 0.35 s without one).
  Every drive is time-limited (<= 10 s); respect the 409 obstacle block. The Pi switches the motors
  OFF 0.75 s without a heartbeat (LINK_TIMEOUT_S), so aisana_legs keeps them on with a 0.15 s keep-alive
  like the Pi's /control page. Owner's decision (Nov 23): automatic calibration is allowed (announced,
  max 2 per 10 min); never by voice.
- Repo lives at `~/ai_robot` on the Jetson; the owner updates with `git pull --ff-only origin main`.
- On the Jetson, system `python3` has no pip. Realtime mode runs in `~/ai_robot/.venv`
  (`.venv/bin/python aisana_realtime.py`); `robot.py` runs with system `python3` (JetPack cv2).

## Code map
- `aisana_realtime.py` — main fast mode (OpenAI Realtime GA over websocket-client 1.3.1).
- `robot.py` — classic STT → chat → TTS loop; imports cv2 and needs OPENAI_API_KEY at import,
  so tests extract its functions via `ast` (see `tests/test_robot.py::isolated`).
- `aisana_follow.py` — "иди за мной": MobileNet-SSD person detector (cv2.dnn) + follow loop; drives only
  through `LegsController.set_command` (lease: the keep-alive stops a drive nobody renews).
- `aisana_arm.py`, `sadam_features.py`, `aisana_memory.py`, `wake_listener.py` are importable
  without hardware and tested directly.

## Workflow
- Run `python3 -m unittest discover -s tests -v` before every push; CI must be green.
- Work on a branch and open a pull request; do not push untested code to `main`.
- Hardware behaviour (audio, camera, serial, latency) cannot be tested in CI — say so and
  ask the owner for the Jetson console log after changes.
- Some OpenAI model names in the code (gpt-6.1-terra, gpt-6-luna, gpt-transcribe,
  gpt-realtime-2.1-mini) may be newer than your knowledge; do not "fix" them without checking docs.
