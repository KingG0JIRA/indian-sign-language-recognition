import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options

# ------------------ MediaPipe Setup ------------------

BaseOptions = base_options.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

landmarker = HandLandmarker.create_from_options(options)

# ------------------ UI COLORS ------------------

RIGHT_COLOR = (0, 200, 255)
LEFT_COLOR  = (255, 140, 0)
TIP_COLOR   = (0, 255, 120)
TEXT_COLOR  = (0, 0, 0)

SKELETON_CONNECTIONS = [
    (0, 5), (5, 9), (9, 13), (13, 17),
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20)
]

# ------------------ UI HELPERS ------------------

def draw_transparent_rect(img, pt1, pt2, color, alpha=0.25):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_label(img, text, pos, color):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x, y = pos
    cv2.rectangle(img, (x, y - h - 8), (x + w + 10, y), color, -1)
    cv2.putText(img, text, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

# ------------------ LANDMARK CLEANING ------------------

def extract_landmarks(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])

def normalize_position(lm):
    return lm - lm[0]

def normalize_scale(lm):
    ref = np.linalg.norm(lm[9])
    return lm if ref == 0 else lm / ref

class LandmarkSmoother:
    def __init__(self, window=5):
        self.buffer = deque(maxlen=window)

    def smooth(self, lm):
        self.buffer.append(lm)
        return np.mean(self.buffer, axis=0)

left_smoother  = LandmarkSmoother()
right_smoother = LandmarkSmoother()

last_left  = None
last_right = None

# ------------------ FEATURE EXTRACTION ------------------

def distance(a, b):
    return np.linalg.norm(a - b)

def angle(a, b, c):
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cosang, -1.0, 1.0))

def extract_features(lm):
    return np.array([
        distance(lm[4], lm[8]),
        distance(lm[8], lm[12]),
        distance(lm[12], lm[16]),
        distance(lm[16], lm[20]),
        distance(lm[0], lm[8]),
        distance(lm[0], lm[12]),
        distance(lm[0], lm[16]),
        distance(lm[0], lm[20]),
        angle(lm[0], lm[5], lm[6]),
        angle(lm[5], lm[6], lm[7]),
        angle(lm[0], lm[9], lm[10]),
        angle(lm[9], lm[10], lm[11]),
        angle(lm[0], lm[13], lm[14]),
        angle(lm[13], lm[14], lm[15]),
        angle(lm[0], lm[17], lm[18]),
        angle(lm[17], lm[18], lm[19])
    ])

ZERO_HAND = np.zeros(16)

# ------------------ FRAME BUFFER ------------------

BUFFER_SIZE = 30
frame_buffer = deque(maxlen=BUFFER_SIZE)

# ------------------ CAMERA ------------------

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 15)

print("Running | ESC to exit")

# ------------------ MAIN LOOP ------------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)

    left_features  = ZERO_HAND.copy()
    right_features = ZERO_HAND.copy()

    detected_left  = False
    detected_right = False

    if result.hand_landmarks and result.handedness:
        for i, hand_landmarks in enumerate(result.hand_landmarks):

            hand_label = result.handedness[i][0].category_name
            hand_label = "Left" if hand_label == "Right" else "Right"

            raw = extract_landmarks(hand_landmarks)
            raw = normalize_position(raw)
            raw = normalize_scale(raw)

            if hand_label == "Right":
                clean = right_smoother.smooth(raw)
                right_features = extract_features(clean)
                last_right = clean
                detected_right = True
                color = RIGHT_COLOR
            else:
                clean = left_smoother.smooth(raw)
                left_features = extract_features(clean)
                last_left = clean
                detected_left = True
                color = LEFT_COLOR

            h, w, _ = frame.shape
            xs = [int(lm.x * w) for lm in hand_landmarks]
            ys = [int(lm.y * h) for lm in hand_landmarks]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            draw_transparent_rect(
                frame,
                (x_min - 12, y_min - 12),
                (x_max + 12, y_max + 12),
                color
            )

            confidence = int(result.handedness[i][0].score * 100)
            draw_label(
                frame,
                f"{hand_label} hand  {confidence}%",
                (x_min - 10, y_min - 15),
                color
            )

            for start, end in SKELETON_CONNECTIONS:
                x1 = int(hand_landmarks[start].x * w)
                y1 = int(hand_landmarks[start].y * h)
                x2 = int(hand_landmarks[end].x * w)
                y2 = int(hand_landmarks[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), color, 3)

            for idx, lm in enumerate(hand_landmarks):
                x = int(lm.x * w)
                y = int(lm.y * h)
                if idx in [4, 8, 12, 16, 20]:
                    cv2.circle(frame, (x, y), 7, TIP_COLOR, -1)
                else:
                    cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)

    # ------------------ FINAL FEATURE VECTOR ------------------
    final_features = np.concatenate([right_features, left_features])  # (32,)

    # ------------------ FRAME BUFFER UPDATE ------------------
    frame_buffer.append(final_features)

    # ------------------ BUFFER READY ------------------
    if len(frame_buffer) == BUFFER_SIZE:
        sequence = np.array(frame_buffer)  # (30, 32)
        print("Sequence ready:", sequence.shape)

        # NEXT STEP:
        # save sequence OR feed to LSTM

    cv2.imshow("Mini project- ISL Translator", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
