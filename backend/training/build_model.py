import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    Dense,
    Embedding,
    LSTM,
    Add,
    Dropout
)
from tensorflow.keras.models import Model


def build_caption_model(
    vocab_size,
    max_length,
    embedding_dim=256,
    lstm_units=256
):
    # Image features extracted by ResNet50
    image_input = Input(
        shape=(2048,),
        name="image_features"
    )

    image_features = Dense(
        embedding_dim,
        activation="relu"
    )(image_input)

    image_features = Dropout(0.3)(image_features)

    # Caption words provided to the LSTM
    caption_input = Input(
        shape=(max_length - 1,),
        name="caption_sequence"
    )

    caption_embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="word_embedding"
    )(caption_input)

    caption_features = LSTM(
        lstm_units,
        name="caption_lstm"
    )(caption_embedding)

    # Combine image and caption features
    combined = Add()([
        image_features,
        caption_features
    ])

    combined = Dense(
        256,
        activation="relu"
    )(combined)

    combined = Dropout(0.3)(combined)

    # Predict the next word
    output = Dense(
        vocab_size,
        activation="softmax",
        name="word_prediction"
    )(combined)

    model = Model(
        inputs=[image_input, caption_input],
        outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model