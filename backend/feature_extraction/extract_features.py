
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input
)
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class ImageFeatureExtractor:

    def __init__(self):
        self.model = ResNet50(
            weights="imagenet",
            include_top=False,
            pooling="avg"
        )

    def extract_feature(self, image_path):
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        return self.model.predict(image, verbose=0)[0]

    def extract_dataset_features(
        self,
        image_folder,
        image_ids,
        output_file,
        batch_size=32
    ):
        all_features = []
        valid_names = []

        total = len(image_ids)

        for start in range(0, total, batch_size):
            batch_ids = image_ids[start:start + batch_size]
            batch_images = []
            batch_names = []

            for image_name in batch_ids:
                image_path = os.path.join(image_folder, image_name)

                if not os.path.exists(image_path):
                    print(f"Missing image: {image_name}")
                    continue

                try:
                    image = load_img(
                        image_path,
                        target_size=(224, 224)
                    )
                    image = img_to_array(image)

                    batch_images.append(image)
                    batch_names.append(image_name)

                except Exception as error:
                    print(f"Could not process {image_name}: {error}")

            if batch_images:
                batch_array = np.array(batch_images)
                batch_array = preprocess_input(batch_array)

                batch_features = self.model.predict(
                    batch_array,
                    verbose=0
                )

                all_features.extend(batch_features)
                valid_names.extend(batch_names)

            processed = min(start + batch_size, total)
            print(f"Processed {processed}/{total} images")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        np.savez_compressed(
            output_file,
            image_names=np.array(valid_names),
            features=np.array(all_features, dtype=np.float32)
        )

        print("\nFEATURE EXTRACTION COMPLETE")
        print("Images processed:", len(valid_names))
        print("Feature shape:", np.array(all_features).shape)
        print("Saved to:", output_file)

        return valid_names, all_features
