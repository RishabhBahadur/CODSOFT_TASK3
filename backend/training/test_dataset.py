import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)

from backend.training.caption_dataset import CaptionDataGenerator

generator = CaptionDataGenerator(
    split_name="train",
    batch_size=64
)

inputs, targets = generator[0]

print("\nCAPTION DATA GENERATOR TEST")
print("----------------------------")
print("Number of batches:", len(generator))
print("Image batch shape:", inputs["image_features"].shape)
print("Caption batch shape:", inputs["caption_sequence"].shape)
print("Target batch shape:", targets.shape)
print("First target token ID:", targets[0])
print("\nData generator test successful!")