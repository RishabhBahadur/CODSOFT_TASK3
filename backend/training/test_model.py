import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)

from backend.training.build_model import build_caption_model

# Temporary values for architecture testing
vocab_size = 5000
max_length = 35

model = build_caption_model(
    vocab_size=vocab_size,
    max_length=max_length
)

print("\nSEE NARY AI - LSTM MODEL")
print("------------------------")

model.summary()

print("\nModel architecture created successfully!")