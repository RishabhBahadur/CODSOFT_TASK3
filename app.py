import os
import uuid
import json

import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.resnet50 import ResNet50

from backend.evaluation.generate_caption import (
    PROJECT_ROOT,
    VOCAB_PATH,
    MODEL_PATH,
    generate_caption,
)


FRONTEND_FOLDER = os.path.join(PROJECT_ROOT, "frontend")
CSS_FOLDER = os.path.join(FRONTEND_FOLDER, "css")
JS_FOLDER = os.path.join(FRONTEND_FOLDER, "js")
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# Load the trained caption model and image encoder once at startup.
print("Loading SeeNary AI caption model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading ResNet50 image encoder...")
encoder = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

with open(VOCAB_PATH, "r", encoding="utf-8") as file:
    vocabulary = json.load(file)

print("SeeNary AI is ready.")


# Serve the frontend page.
@app.route("/")
def home():
    return send_from_directory(FRONTEND_FOLDER, "index.html")


# Serve the separate CSS and JavaScript files.
@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(CSS_FOLDER, filename)


@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory(JS_FOLDER, filename)


# Receive an uploaded image and return its generated caption.
@app.route("/api/generate", methods=["POST"])
def generate():
    if "image" not in request.files:
        return jsonify({"error": "Please select an image."}), 400

    image_file = request.files["image"]

    if not image_file or image_file.filename == "":
        return jsonify({"error": "Please select an image."}), 400

    if not allowed_file(image_file.filename):
        return jsonify({
            "error": "Unsupported image type. Use JPG, JPEG, PNG, or WEBP."
        }), 400

    safe_name = secure_filename(image_file.filename)

    if "." not in safe_name:
        return jsonify({"error": "Invalid image filename."}), 400

    extension = safe_name.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{extension}"
    image_path = os.path.join(UPLOAD_FOLDER, unique_name)

    try:
        image_file.save(image_path)

        caption = generate_caption(
            model=model,
            encoder=encoder,
            vocabulary=vocabulary,
            image_path=image_path
        )

        return jsonify({"caption": caption})

    except Exception:
        app.logger.exception("Caption generation failed")
        return jsonify({
            "error": "Could not generate a caption for this image."
        }), 500

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "error": "Image is too large. Maximum size is 16 MB."
    }), 413


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)