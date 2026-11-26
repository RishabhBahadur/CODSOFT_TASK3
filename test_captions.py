
from backend.preprocessing.caption_preprocessing import (
    load_captions,
    load_image_ids,
    build_vocabulary,
    get_max_caption_length
)

CAPTION_FILE = "dataset/captions/Flickr8k.token.txt"
TRAIN_FILE = "dataset/captions/Flickr_8k.trainImages.txt"

captions = load_captions(CAPTION_FILE)
train_ids = load_image_ids(TRAIN_FILE)

vocabulary = build_vocabulary(captions, train_ids)
max_length = get_max_caption_length(captions, train_ids)

print("CAPTION PREPROCESSING SUCCESSFUL")
print("--------------------------------")
print("Total images with captions:", len(captions))
print("Training images:", len(train_ids))
print("Vocabulary size:", len(vocabulary))
print("Maximum caption length:", max_length)

sample_image = train_ids[0]
print("\nSample image:", sample_image)
print("Sample captions:")

for caption in captions.get(sample_image, []):
    print("-", caption)
