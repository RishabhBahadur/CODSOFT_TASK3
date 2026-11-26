
import numpy as np
import os

files = [
    "models/train_features.npz",
    "models/validation_features.npz",
    "models/test_features.npz"
]

for file_path in files:
    if os.path.exists(file_path):
        data = np.load(file_path)
        print(os.path.basename(file_path))
        print("Feature shape:", data["features"].shape)
        print("Image count:", len(data["image_names"]))
        print("--------------------------")
    else:
        print("Missing:", file_path)
