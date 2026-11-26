
import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)

from backend.feature_extraction.extract_features import ImageFeatureExtractor
from backend.preprocessing.caption_preprocessing import load_image_ids


CAPTIONS_DIR = os.path.join(PROJECT_ROOT, "dataset", "captions")
IMAGE_DIR = os.path.join(PROJECT_ROOT, "dataset", "images")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

SPLITS = {
    "train": "Flickr_8k.trainImages.txt",
    "validation": "Flickr_8k.devImages.txt",
    "test": "Flickr_8k.testImages.txt"
}


def main():
    extractor = ImageFeatureExtractor()

    for split_name, split_file in SPLITS.items():
        print(f"\nExtracting {split_name} features...")

        split_path = os.path.join(CAPTIONS_DIR, split_file)
        image_ids = load_image_ids(split_path)

        output_path = os.path.join(
            MODEL_DIR,
            f"{split_name}_features.npz"
        )

        extractor.extract_dataset_features(
            IMAGE_DIR,
            image_ids,
            output_path,
            batch_size=32
        )


if __name__ == "__main__":
    main()
