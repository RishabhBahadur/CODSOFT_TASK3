SeeNary AI



An image captioning application that generates a natural-language description of an uploaded image.



FEATURES

- Upload an image through a web interface

- Generate a caption using a trained deep-learning model

- Flask backend with an HTML, CSS, and JavaScript frontend

MODEL



The captioning pipeline uses ResNet50 for image feature extraction and an LSTM-based decoder to generate captions.



The application requires these files to run:

- models/best_caption_model.keras - trained captioning model

- models/vocabulary.json - vocabulary used by the caption generator

The Flickr8k dataset and generated feature files are not included in this repository.



SETUP

1. Install the dependencies by running: python -m pip install -r requirements.txt

2. Make sure the trained model and vocabulary files are present in the models folder.

3. Start the application by running: python app.py

4. Open the local address printed in the terminal in your browser.

NOTE



This is an educational image captioning prototype. Generated captions may be inaccurate or incomplete.


