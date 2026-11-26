
import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)

from backend.feature_extraction.extract_features import ImageFeatureExtractor

image_path = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "images",
    "1001773457_577c3a7d70.jpg"
)

print("Loading ResNet50...")
extractor = ImageFeatureExtractor()

print("Extracting image features...")
features = extractor.extract_feature(image_path)

print("\nRESNET50 TEST SUCCESSFUL")
print("------------------------")
print("Feature shape:", features.shape)
print("Feature dimensions:", len(features))
print("Sample feature values:", features[:10])
