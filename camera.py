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
    # Small JPEG keeps vision uploads reliable on slow university Wi-Fi.\n    frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)\n    cv2.imwrite("photo.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
    print("Photo saved: photo.jpg")
else:
    print("Failed to capture frame")

camera.release()
