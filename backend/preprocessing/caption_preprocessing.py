
import re
from collections import Counter


def clean_caption(caption):
    """Clean a caption and add start/end sequence markers."""
    caption = caption.lower()
    caption = re.sub(r"[^a-z\s]", " ", caption)
    words = caption.split()
    words = [word for word in words if len(word) > 1]

    return "startseq " + " ".join(words) + " endseq"


def load_captions(caption_file):
    """Load captions and group them by image filename."""
    captions = {}

    with open(caption_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            image_info, separator, caption = line.partition("\t")

            if not separator:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                image_info, caption = parts

            image_name = image_info.split("#")[0]
            cleaned = clean_caption(caption)

            captions.setdefault(image_name, []).append(cleaned)

    return captions


def load_image_ids(split_file):
    """Load image filenames from a train, validation, or test split."""
    with open(split_file, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def build_vocabulary(captions, image_ids, min_word_freq=1):
    """Build vocabulary using only captions from selected images."""
    word_counts = Counter()

    for image_name in image_ids:
        for caption in captions.get(image_name, []):
            word_counts.update(caption.split())

    vocabulary = [
        word for word, count in word_counts.items()
        if count >= min_word_freq
    ]

    vocabulary = ["<pad>", "<unk>"] + sorted(vocabulary)

    word_to_index = {
        word: index for index, word in enumerate(vocabulary)
    }

    return word_to_index


def get_max_caption_length(captions, image_ids):
    """Find the maximum caption length for selected images."""
    lengths = []

    for image_name in image_ids:
        for caption in captions.get(image_name, []):
            lengths.append(len(caption.split()))

    return max(lengths) if lengths else 0
