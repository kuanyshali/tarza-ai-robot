import os

import cv2

pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), "
    "width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false"
)

camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not camera.isOpened():
    raise RuntimeError("Camera initialization failed")

ret, frame = camera.read()

if ret:
    # Small JPEG keeps vision uploads reliable on slow university Wi-Fi.
    # The camera is mounted upside down; same setting as robot.py.
    rotation = {
        "90": cv2.ROTATE_90_CLOCKWISE,
        "180": cv2.ROTATE_180,
        "270": cv2.ROTATE_90_COUNTERCLOCKWISE,
    }.get(os.environ.get("TARZA_CAMERA_ROTATE", "180").strip())
    if rotation is not None:
        frame = cv2.rotate(frame, rotation)
    frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite("photo.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]):
        raise RuntimeError("Failed to save photo.jpg")
    print("Photo saved: photo.jpg")
else:
    print("Failed to capture frame")

camera.release()
