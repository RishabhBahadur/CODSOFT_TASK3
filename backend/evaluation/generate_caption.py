import os
import sys
import json
import numpy as np
import tensorflow as tf



from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "best_caption_model.keras"
)

VOCAB_PATH = os.path.join(
    PROJECT_ROOT, "models", "vocabulary.json"
)


def extract_image_features(image_path, encoder):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    features = encoder.predict(image, verbose=0)
    return features


def generate_caption(model, encoder, vocabulary, image_path):
    word_to_index = vocabulary["word_to_index"]
    max_length = vocabulary["max_length"]

    index_to_word = {
        int(index): word
        for word, index in word_to_index.items()
    }

    if "startseq" not in word_to_index:
        raise ValueError("The vocabulary does not contain 'startseq'.")

    features = extract_image_features(image_path, encoder)
    sequence = [word_to_index["startseq"]]

    for _ in range(max_length - 1):
        padded_sequence = pad_sequences(
            [sequence],
            maxlen=max_length - 1,
            padding="post",
            truncating="post",
            value=word_to_index["<pad>"]
        )

        predictions = model(
    {
        "image_features": features,
        "caption_sequence": padded_sequence
    },
    training=False
).numpy()

        next_index = int(np.argmax(predictions[0]))
        next_word = index_to_word.get(next_index)

        if next_word is None or next_word == "endseq":
            break

        sequence.append(next_index)

    generated_words = [
        index_to_word[index]
        for index in sequence[1:]
        if index in index_to_word
    ]

    return " ".join(generated_words)


def main():
    if len(sys.argv) < 2:
        print(
            'Usage: python backend\\evaluation\\generate_caption.py '
            '"dataset\\images\\your_image.jpg"'
        )
        return

    image_path = os.path.abspath(sys.argv[1])

    if not os.path.isfile(image_path):
        print("Image not found:", image_path)
        return

    with open(VOCAB_PATH, "r", encoding="utf-8") as file:
        vocabulary = json.load(file)

    print("Loading caption model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Loading ResNet50 encoder...")
    encoder = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg"
    )

    print("Generating caption...")
    caption = generate_caption(
        model=model,
        encoder=encoder,
        vocabulary=vocabulary,
        image_path=image_path
    )

    print("\nGenerated caption:", caption)


if __name__ == "__main__":
    main()