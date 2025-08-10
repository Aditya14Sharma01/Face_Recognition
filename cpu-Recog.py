import cv2
import time
import numpy as np
from deepface import DeepFace
from threading import Thread

# Global variables
latest_results = []
last_analysis_time = 0
REFRESH_INTERVAL = 3  # seconds
TF_ENABLE_ONEDNN_OPTS = 0

# Choose the backend
DETECTOR_BACKEND = "retinaface"

# Preload the DeepFace models (warm-up)
print("[INFO] Loading DeepFace models... Please wait...")
_ = DeepFace.analyze(
    np.zeros((720, 1280, 3), dtype=np.uint8),
    actions=['age', 'gender', 'emotion'],
    detector_backend=DETECTOR_BACKEND,
    enforce_detection=False
)
print("[INFO] Models loaded successfully.")

def analyze_frame(frame):
    global latest_results
    try:
        results = DeepFace.analyze(
            frame,
            actions=['age', 'gender', 'emotion'],
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False
        )
        if not isinstance(results, list):
            results = [results]
        latest_results = results
    except Exception as e:
        print("Detection error:", e)

# Start camera
cap = cv2.VideoCapture(0)

# Set high resolution for better accuracy
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Create resizable window
window_name = "Human Face Recognition"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror view

    # Background AI analysis every REFRESH_INTERVAL seconds
    if time.time() - last_analysis_time > REFRESH_INTERVAL:
        last_analysis_time = time.time()
        Thread(target=analyze_frame, args=(frame.copy(),), daemon=True).start()

    # Draw last detection results
    for person in latest_results:
        try:
            x, y, w, h = (
                person['region']['x'],
                person['region']['y'],
                person['region']['w'],
                person['region']['h']
            )
            x, y = max(x, 0), max(y, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            label = f"{person['dominant_gender']}, {person['age']} yrs, {person['dominant_emotion']}"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        except:
            pass

    # Stretch to window size
    win_width = int(cv2.getWindowImageRect(window_name)[2])
    win_height = int(cv2.getWindowImageRect(window_name)[3])
    if win_width > 0 and win_height > 0:
        frame = cv2.resize(frame, (win_width, win_height))

    cv2.imshow(window_name, frame)

    # Exit if 'q' pressed or window closed
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
