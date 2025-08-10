import cv2
from deepface import DeepFace
import torch
import os
import warnings

# Hide warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# GPU info
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_count = 0
results_cache = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image for natural view
    frame = cv2.flip(frame, 1)

    # Slight brightness/contrast boost
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)

    frame_count += 1

    if frame_count % 10 == 0:
        try:
            results = DeepFace.analyze(
                frame,
                actions=['age', 'gender', 'emotion'],
                detector_backend='retinaface',  # More accurate than opencv
                enforce_detection=False
            )
            if isinstance(results, dict):
                results_cache = [results]
            elif isinstance(results, list):
                results_cache = results
        except Exception as e:
            print("Detection error:", e)

    # Draw results
    if results_cache:
        for r in results_cache:
            face_area = r.get('facial_area')
            if face_area:
                x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']

                # Thick cyan rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)

                # Info text
                info_text = f"Age: {r.get('age', '?')}  Gender: {r.get('gender', '?')}  Emotion: {r.get('dominant_emotion', '?')}"
                (text_w, text_h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

                # Cyan background above the face
                cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 10, y), (255, 255, 0), -1)
                cv2.putText(frame, info_text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Show window
    cv2.imshow("Face Analysis", frame)

    # Close on ESC or X button
    if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty("Face Analysis", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
