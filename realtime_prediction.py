import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model
import json
import re
import time

# Config
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.7
HISTORY_LENGTH = 5
FRAME_SKIP = 2
SENTENCE_HOLD_DURATION = 5  # seconds

# State variables
sequence = deque(maxlen=SEQUENCE_LENGTH)
prediction_history = deque(maxlen=HISTORY_LENGTH)
words_collected = []
predicted_text = "Show your hand..."
last_sentence_time = 0
sentence_frozen = False

# Rule-based sentence matching
def clean_sentence(words):
    text = ' '.join(words).lower()
    patterns = [
        (r'what you name', "What is your name?"),
        (r'you fine', "Are you fine?"),
        (r'i fine', "I am fine."),
        (r'i want food', "I want food."),
        (r'where you', "Where are you?"),
        (r'i help', "I need Help!"),
        (r'what you want', "Do you need something?"),
    ]
    for pattern, response in patterns:
        if re.search(pattern, text):
            return response
    return None

# Load model and label map
model = load_model("models/lstm_model.keras")
with open("models/label_map.json") as f:
    label_map = json.load(f)
reverse_map = {v: k for k, v in label_map.items()}
print("Label Map Verification:")
print(f"Sample mapping: 0 -> {reverse_map.get(0, 'Unknown')}")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Normalize and extract keypoints
def normalize_landmarks(landmarks):
    if not landmarks:
        return [(0.0, 0.0)] * 21
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    wrist_x, wrist_y = x_coords[0], y_coords[0]
    x_coords = [x - wrist_x for x in x_coords]
    y_coords = [y - wrist_y for y in y_coords]
    scale = max(max(map(abs, x_coords + y_coords)), 1e-6)
    return [(x / scale, y / scale) for x, y in zip(x_coords, y_coords)]

def extract_keypoints(results):
    if not results.multi_hand_landmarks:
        return np.zeros(84)
    hand_data = [None, None]
    if results.multi_handedness:
        handedness = [h.classification[0].label for h in results.multi_handedness]
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            idx = 0 if handedness[i] == 'Right' else 1
            hand_data[idx] = normalize_landmarks(hand_landmarks.landmark)
    for i in range(2):
        if hand_data[i] is None:
            hand_data[i] = [(0.0, 0.0)] * 21
    flattened = [coord for hand in hand_data for point in hand for coord in point]
    return np.array(flattened)

# Main loop
cap = cv2.VideoCapture(0)
frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame_counter += 1
    if frame_counter % FRAME_SKIP != 0:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    hand_detected = False
    if results.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)
            )

    if not hand_detected and len(sequence) > 0:
        sequence.clear()
        prediction_history.clear()
        continue

    keypoints = extract_keypoints(results)
    sequence.append(keypoints)

    if len(sequence) == SEQUENCE_LENGTH and hand_detected and not sentence_frozen:
        try:
            input_data = np.expand_dims(np.array(sequence), axis=0)
            prediction = model.predict(input_data, verbose=0)[0]
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]

            if confidence > CONFIDENCE_THRESHOLD:
                prediction_history.append(predicted_class)

                if len(prediction_history) == HISTORY_LENGTH:
                    final_prediction = max(set(prediction_history), key=list(prediction_history).count)
                    predicted_word = reverse_map.get(int(final_prediction), "Unknown")

                    if not words_collected or words_collected[-1] != predicted_word:
                        words_collected.append(predicted_word)
                        print(f"Detected word: {predicted_word}")

                    matched = clean_sentence(words_collected)
                    if matched:
                        predicted_text = matched
                        sentence_frozen = True
                        last_sentence_time = time.time()
                        print(f"Matched sentence: {predicted_text}")

        except Exception as e:
            print(f"Prediction error: {str(e)}")

    # Unfreeze after duration
    if sentence_frozen and (time.time() - last_sentence_time) > SENTENCE_HOLD_DURATION:
        sentence_frozen = False
        words_collected.clear()
        prediction_history.clear()
        predicted_text = "Show your hand..."

    # UI Display
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (50, 50, 50), -1)
    cv2.rectangle(frame, (0, frame.shape[0] - 40), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1)

    cv2.putText(frame, predicted_text, (20, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    buffer_status = f"Buffer: {len(sequence)}/{SEQUENCE_LENGTH}"
    cv2.putText(frame, buffer_status, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    status = "Hand: Detected" if hand_detected else "Hand: Searching..."
    cv2.putText(frame, status, (frame.shape[1] - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    if len(sequence) == SEQUENCE_LENGTH and hand_detected and 'confidence' in locals():
        conf_text = f"Confidence: {confidence:.2f}"
        cv2.putText(frame, conf_text, (frame.shape[1] - 200, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    cv2.imshow("ASL Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
