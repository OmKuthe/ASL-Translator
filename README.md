# 🤟 ASL Translator – Real-Time Sign Language Recognition

A real-time American Sign Language (ASL) translator that converts sign gestures into complete English sentences using MediaPipe for hand tracking and gesture classification. The system implements rule-based logic to form grammatically correct sentences from recognized gestures, making it an intuitive communication tool for basic ASL interpretation.

---

## ✨ Key Features

### Real-Time Processing
- 🖐️ Accurate hand tracking using MediaPipe's Hand Landmarks
- ⚡ Smooth webcam-based gesture detection
- 🎭 Background-agnostic detection (works in varied environments)

### Gesture Recognition
- 🔄 Sequence buffer (30-frame window) for stable recognition
- 📊 Confidence thresholding to ensure reliable predictions
- ✨ Normalized coordinates for position/scale invariance

### User Experience
- 📝 Dynamic sentence construction with rule-based grammar
- 🎯 Visual feedback (prediction buffer, status indicators)
- ⏳ Auto-clearing of displayed sentences after delay
- 🎨 Clean UI overlay with OpenCV visualization

---

## 🛠️ Technical Implementation

### Core Technologies

| Component          | Technology Used     |
|--------------------|---------------------|
| Hand Tracking      | MediaPipe Hands     |
| Classification     | Keras Model (Dense) |
| Computer Vision    | OpenCV              |
| Data Processing    | NumPy, JSON         |

> 📌 Note: This project uses a **frame-wise gesture classifier** trained on normalized keypoints — not an LSTM-based sequence model.

---

## 📋 Supported Gestures & Sentences

### Recognized Vocabulary (15 Signs)

["what", "your", "name", "my", "is", 
 "how", "you", "are", "fine", "where",
 "help", "please", "I", "want", "food"]
### Smart Sentence Formation

| Gesture Sequence       | Output Sentence          |
|------------------------|--------------------------|
| what → you → name      | What is your name?       |
| you → fine             | Are you fine?            |
| I → fine               | I am fine.               |
| I → want → food        | I want food.             |
| where → you            | Where are you?           |
| I → help               | I need help!             |
| what → you → want      | Do you need something?   |

---

### 🚀 Getting Started

### Installation

```bash
pip install opencv-python mediapipe tensorflow matplotlib scikit-learn
```

## Workflow

### 📦 Data Collection

```bash
python data_collection.py
```
Press number keys (0–9) and letters (a–e) to record each gesture

Saves data to data/X.npy and data/y.npy

🧠 Model Training
```bash
python model_training.py
```
Trains the gesture classifier

Saves model as models/lstm_model.keras (name kept for compatibility)

🎥 Real-Time Translation
```bash
python predict_realtime.py
Uses webcam feed to detect gestures and display sentences
```
📂 Project Structure

```bash
ASL-Translator/
├── data/                   # Collected gesture datasets
│   ├── X.npy               # Feature vectors
│   └── y.npy               # Corresponding labels
├── models/                 # Trained model & label maps
│   ├── lstm_model.keras    # Gesture classifier
│   └── label_map.json      # Gesture-to-index mapping
├── data_collection.py      # Gesture recording script
├── model_training.py       # Model training pipeline
├── predict_realtime.py     # Live translation system
└── README.md               # Documentation
```
