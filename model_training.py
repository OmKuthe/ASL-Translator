import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
import json

# Load data
X = np.load("data/X.npy")  # shape: (samples, 30, 84)
y = np.load("data/y.npy")

# Label processing
y = np.round(y).astype(int)
classes = np.unique(y)
label_map = {int(label): int(idx) for idx, label in enumerate(classes)}

# Save label mapping with gesture names for reference
os.makedirs("models", exist_ok=True)
with open("models/label_map.json", "w") as f:
    json.dump(label_map, f)

# Create reverse mapping with gesture names
gesture_names = [
    "what", "your", "name", "my", "is",
    "how", "you", "are", "fine", "where",
    "help", "please", "I", "want", "food"
]
name_map = {idx: gesture_names[label] for label, idx in label_map.items()}
with open("models/gesture_names.json", "w") as f:
    json.dump(name_map, f)

y = np.array([label_map[val] for val in y])
print("Sanitized labels:", np.unique(y))
print("Label distribution:", np.bincount(y))

num_classes = len(np.unique(y))
y_cat = to_categorical(y, num_classes=num_classes)

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
)

# Enhanced LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, activation='tanh', input_shape=(30, 84)),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64, return_sequences=False, activation='tanh'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Custom learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy'),
    ModelCheckpoint("models/best_lstm.keras", save_best_only=True, monitor='val_accuracy')
]

# Enhanced training
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# Load best model
model = load_model("models/best_lstm.keras")

# Evaluation
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n[RESULT] Classification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=[name_map[i] for i in range(num_classes)]))

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show()

# Save final model
model.save("models/lstm_model.keras")
print("[INFO] Model saved as 'models/lstm_model.keras'")

# Save prediction test samples (for debugging)
np.save("models/test_samples.npy", X_test[:10])
print("[INFO] Saved test samples for verification")