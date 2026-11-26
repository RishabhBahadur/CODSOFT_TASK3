import os
import sys
import json

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)

from backend.preprocessing.caption_preprocessing import (
    load_captions,
    load_image_ids,
    build_vocabulary,
    get_max_caption_length
)

CAPTION_FILE = os.path.join(
    PROJECT_ROOT, "dataset", "captions", "Flickr8k.token.txt"
)

TRAIN_FILE = os.path.join(
    PROJECT_ROOT, "dataset", "captions", "Flickr_8k.trainImages.txt"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT, "models", "vocabulary.json"
)


def main():
    print("Loading captions and training image list...")

    captions = load_captions(CAPTION_FILE)
    train_ids = load_image_ids(TRAIN_FILE)

    print("Building vocabulary from training captions...")
    word_to_index = build_vocabulary(captions, train_ids)

    max_length = get_max_caption_length(captions, train_ids)

    vocabulary_data = {
        "word_to_index": word_to_index,
        "vocab_size": len(word_to_index),
        "max_length": max_length
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(vocabulary_data, file, ensure_ascii=False, indent=2)

    print("\nVOCABULARY PREPARATION COMPLETE")
    print("Training images:", len(train_ids))
    print("Vocabulary size:", len(word_to_index))
    print("Maximum caption length:", max_length)
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()