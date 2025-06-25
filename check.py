import numpy as np

X = np.load("data/X.npy")
y = np.load("data/y.npy")

print("X shape:", X.shape)  # should be (samples, 30, 84)
print("y shape:", y.shape)  # should be (samples,)

print("Labels example:", y[:5])
print("One sample shape:", X[0].shape)
print("First frame of sample 0:", X[0][0])  # shape (84,)
