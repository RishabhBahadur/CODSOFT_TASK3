
import os
import sys
import json
import shutil
import tensorflow as tf

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.insert(0, PROJECT_ROOT)

from backend.training.build_model import build_caption_model
from backend.training.caption_dataset import CaptionDataGenerator


def main():

    # Load vocabulary information
    vocabulary_path = os.path.join(
        PROJECT_ROOT,
        "models",
        "vocabulary.json"
    )

    with open(vocabulary_path, "r", encoding="utf-8") as file:
        vocabulary_data = json.load(file)

    vocab_size = vocabulary_data["vocab_size"]
    max_length = vocabulary_data["max_length"]

    print("Vocabulary size:", vocab_size)
    print("Maximum caption length:", max_length)

    # Model paths
    checkpoint_path = os.path.join(
        PROJECT_ROOT,
        "models",
        "best_caption_model.keras"
    )

    backup_path = os.path.join(
        PROJECT_ROOT,
        "models",
        "best_caption_model_backup.keras"
    )

    # Protect the previously trained model
    if os.path.exists(checkpoint_path):

        if not os.path.exists(backup_path):
            shutil.copy2(checkpoint_path, backup_path)
            print("\nExisting model backup created.")

        else:
            print("\nExisting model backup already available.")

    # Create training generator
    print("\nLoading training dataset...")

    train_generator = CaptionDataGenerator(
        split_name="train",
        batch_size=128,
        shuffle=True
    )

    # Create validation generator
    print("\nLoading validation dataset...")

    validation_generator = CaptionDataGenerator(
        split_name="validation",
        batch_size=128,
        shuffle=False
    )

    print("\nTraining batches:", len(train_generator))
    print("Validation batches:", len(validation_generator))

    # Build caption model
    print("\nBuilding caption model...")

    model = build_caption_model(
        vocab_size=vocab_size,
        max_length=max_length
    )

    model.summary()

    # Save the best model during this training run
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )

    # Stop training if validation loss stops improving
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1
    )

    # Reduce learning rate when validation loss plateaus
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=0.00001,
        verbose=1
    )

    callbacks = [
        checkpoint,
        early_stopping,
        reduce_lr
    ]

    # Train model
    print("\nStarting caption model training...")

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=5,
        callbacks=callbacks,
        verbose=1
    )

    print("\nTRAINING COMPLETED")

    best_val_loss = min(history.history["val_loss"])
    best_val_accuracy = max(history.history["val_accuracy"])

    print("Best validation loss:", round(best_val_loss, 4))
    print("Best validation accuracy:", round(best_val_accuracy, 4))

    print("\nBest model saved at:")
    print(checkpoint_path)

    print("\nPrevious model backup:")
    print(backup_path)


if __name__ == "__main__":
    main()
