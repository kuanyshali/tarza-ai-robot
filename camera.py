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
    cv2.imwrite("photo.jpg", frame)
    print("Photo saved: photo.jpg")
else:
    print("Failed to capture frame")

camera.release()
