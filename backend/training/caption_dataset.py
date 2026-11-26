import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)

from backend.preprocessing.caption_preprocessing import (
    load_captions,
    load_image_ids,
)


class CaptionDataGenerator(tf.keras.utils.Sequence):

    def __init__(
        self,
        split_name,
        batch_size=64,
        shuffle=True,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.shuffle = shuffle

        captions_path = os.path.join(
            PROJECT_ROOT,
            "dataset",
            "captions",
            "Flickr8k.token.txt",
        )

        split_files = {
            "train": "Flickr_8k.trainImages.txt",
            "validation": "Flickr_8k.devImages.txt",
            "test": "Flickr_8k.testImages.txt",
        }

        if split_name not in split_files:
            raise ValueError(
                f"Invalid split_name: {split_name}. "
                f"Choose from {list(split_files.keys())}."
            )

        split_path = os.path.join(
            PROJECT_ROOT,
            "dataset",
            "captions",
            split_files[split_name],
        )

        feature_path = os.path.join(
            PROJECT_ROOT,
            "models",
            f"{split_name}_features.npz",
        )

        vocabulary_path = os.path.join(
            PROJECT_ROOT,
            "models",
            "vocabulary.json",
        )

        with open(vocabulary_path, "r", encoding="utf-8") as file:
            vocabulary_data = json.load(file)

        self.word_to_index = vocabulary_data["word_to_index"]
        self.max_length = vocabulary_data["max_length"]

        self.pad_index = self.word_to_index["<pad>"]
        self.unk_index = self.word_to_index["<unk>"]

        feature_data = np.load(feature_path)

        self.image_features = {
            name: feature
            for name, feature in zip(
                feature_data["image_names"],
                feature_data["features"],
            )
        }

        captions = load_captions(captions_path)
        image_ids = load_image_ids(split_path)

        self.samples = []

        for image_name in image_ids:
            if image_name not in self.image_features:
                continue

            for caption in captions.get(image_name, []):
                token_ids = [
                    self.word_to_index.get(word, self.unk_index)
                    for word in caption.split()
                ]

                # Each next-word prediction becomes one training sample.
                for position in range(1, len(token_ids)):
                    self.samples.append(
                        (image_name, token_ids, position)
                    )

        self.indices = np.arange(len(self.samples))
        self.on_epoch_end()

        print(f"{split_name.title()} samples:", len(self.samples))

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, batch_index):
        batch_indices = self.indices[
            batch_index * self.batch_size:
            (batch_index + 1) * self.batch_size
        ]

        current_batch_size = len(batch_indices)
        sequence_length = self.max_length - 1

        image_batch = np.zeros(
            (current_batch_size, 2048),
            dtype=np.float32,
        )

        caption_batch = np.full(
            (current_batch_size, sequence_length),
            fill_value=self.pad_index,
            dtype=np.int32,
        )

        target_batch = np.zeros(
            (current_batch_size,),
            dtype=np.int32,
        )

        for row, sample_index in enumerate(batch_indices):
            image_name, token_ids, position = self.samples[sample_index]

            image_batch[row] = self.image_features[image_name]

            # Input contains the caption prefix; target is the next word.
            prefix = token_ids[:position]
            prefix = prefix[:sequence_length]

            padded_prefix = pad_sequences(
                [prefix],
                maxlen=sequence_length,
                padding="post",
                truncating="post",
                value=self.pad_index,
            )[0]

            caption_batch[row] = padded_prefix
            target_batch[row] = token_ids[position]

        inputs = {
            "image_features": image_batch,
            "caption_sequence": caption_batch,
        }

        return inputs, target_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)