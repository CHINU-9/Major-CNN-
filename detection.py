import cv2
import numpy as np
from datetime import datetime
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ---------------- MODEL ----------------
cnn = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

model = Sequential([
    LSTM(128, input_shape=(16, 1280)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')

# ---------------- TIME FUNCTION ----------------
def frame_to_time(frame, fps):
    seconds = int(frame / fps)
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224,224))
        frame = frame / 255.0

        feat = cnn.predict(np.expand_dims(frame, axis=0), verbose=0)
        frames.append(feat[0])

    cap.release()
    return frames

# ---------------- ACCIDENT DETECTION ----------------
def detect_accident(features):
    accident_frames = []

    for i in range(1, len(features)):
        diff = np.linalg.norm(features[i] - features[i-1])

        if diff > 100:   # threshold
            accident_frames.append(i)

    return accident_frames

# ---------------- SAVE CLIP ----------------
def save_clip(video_path, frame_id):
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start = max(0, frame_id - 2*fps)
    end = min(total_frames, frame_id + 3*fps)

    filename = datetime.now().strftime("%d-%m-%y_%H-%M") + ".mp4"

    out = cv2.VideoWriter(filename,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (w, h))

    for f in range(start, end):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)

        ret, frame = cap.read()
        if not ret:
            continue

        video_time = frame_to_time(f, fps)
        system_time = datetime.now().strftime("%d-%m-%y %H:%M:%S")

        cv2.putText(frame, "ACCIDENT DETECTED", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.putText(frame, f"Video Time: {video_time}", (50,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.putText(frame, f"System Time: {system_time}", (50,140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        out.write(frame)

    out.release()
    cap.release()

    return filename