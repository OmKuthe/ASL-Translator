import cv2
import mediapipe as mp
import numpy as np
import os
from collections import defaultdict

# Enhanced setup
GESTURES = [
    "what", "your", "name", "my", "is",
    "how", "you", "are", "fine", "where",
    "help", "please", "I", "want", "food"
]
FRAMES_PER_SAMPLE = 30
MIN_SAMPLES_PER_CLASS = 30

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# Data storage
samples_collected = defaultdict(int)
sequence = []
X = []
y = []


# Normalization function
def normalize(hand_landmarks):
    if not hand_landmarks:
        return [(0.0, 0.0)] * 21

    x_coords = [lm.x for lm in hand_landmarks]
    y_coords = [lm.y for lm in hand_landmarks]
    wrist_x, wrist_y = x_coords[0], y_coords[0]
    x_coords = [x - wrist_x for x in x_coords]
    y_coords = [y - wrist_y for y in y_coords]
    scale = max(max(map(abs, x_coords + y_coords)), 1e-6)
    return [(x / scale, y / scale) for x, y in zip(x_coords, y_coords)]


# Key mapping for gestures (0-9 plus additional keys)
KEY_MAPPING = {
    ord('0'): 0, ord('1'): 1, ord('2'): 2, ord('3'): 3, ord('4'): 4,
    ord('5'): 5, ord('6'): 6, ord('7'): 7, ord('8'): 8, ord('9'): 9,
    ord('a'): 10, ord('b'): 11, ord('c'): 12, ord('d'): 13, ord('e'): 14
}

cap = cv2.VideoCapture(0)
current_label = None
collecting = False

print("[INFO] Available gestures and keys:")
for i, gesture in enumerate(GESTURES):
    key = [k for k, v in KEY_MAPPING.items() if v == i][0]
    print(f"{chr(key)} : {gesture}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    frame_landmarks = []
    hand_visible = False

    if results.multi_hand_landmarks:
        hand_visible = True
        hand_data = [None, None]

        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[hand_idx].classification[0].label
            idx = 0 if handedness == "Right" else 1
            hand_data[idx] = normalize(hand_landmarks.landmark)

        for i in range(2):
            if hand_data[i] is None:
                hand_data[i] = normalize(None)

        both_hands = hand_data[0] + hand_data[1]
        frame_landmarks = [coord for point in both_hands for coord in point]

    # Draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Data collection
    if collecting and hand_visible:
        sequence.append(frame_landmarks)

        if len(sequence) == FRAMES_PER_SAMPLE:
            X.append(sequence)
            y.append(current_label)
            samples_collected[current_label] += 1
            print(f"[SAVED] Sample {samples_collected[current_label]} for '{GESTURES[current_label]}'")
            sequence = []

            if samples_collected[current_label] >= MIN_SAMPLES_PER_CLASS:
                print(f"[INFO] Collected enough samples for {GESTURES[current_label]}")
                collecting = False

    # Display status
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (50, 50, 50), -1)
    status = f"Collecting: {GESTURES[current_label]} ({samples_collected[current_label]}/{MIN_SAMPLES_PER_CLASS})" if collecting else "Press keys to select gesture"
    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Show collection progress
    if collecting:
        progress = len(sequence) / FRAMES_PER_SAMPLE
        cv2.rectangle(frame, (10, 50), (10 + int(200 * progress), 70), (0, 255, 0), -1)

    cv2.imshow("ASL Data Collector", frame)
    key = cv2.waitKey(10)

    if key == ord('q'):
        break
    elif key in KEY_MAPPING:
        current_label = KEY_MAPPING[key]
        if samples_collected[current_label] < MIN_SAMPLES_PER_CLASS:
            collecting = True
            sequence = []
            print(f"[START] Collecting samples for '{GESTURES[current_label]}'")
        else:
            print(f"[SKIP] Already have {MIN_SAMPLES_PER_CLASS} samples for '{GESTURES[current_label]}'")
    elif key == ord('s'):
        collecting = False
        sequence = []

# Save data
if len(X) > 0:
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    os.makedirs("data", exist_ok=True)
    np.save("data/X.npy", X)
    np.save("data/y.npy", y)

    print("\n[SUMMARY] Collected samples per class:")
    for i, gesture in enumerate(GESTURES):
        print(f"{gesture}: {samples_collected[i]}")

    print(f"\n[SUCCESS] Saved {len(X)} samples to:")
    print(f"- data/X.npy (shape: {X.shape})")
    print(f"- data/y.npy (shape: {y.shape})")
else:
    print("[WARNING] No data collected!")

cap.release()
cv2.destroyAllWindows()