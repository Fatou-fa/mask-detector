import tensorflow as tf
import numpy as np
import json
import base64
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    data = request.get_json()
    img_data = data.get("image", "")
    if "," in img_data:
        img_data = img_data.split(",")[1]
    try:
        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes))
        img = img.convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)[0]
        confidences = [
            {"label": labels_fr[idx_to_class[str(i)]],
             "confidence": float(predictions[i])}
            for i in range(len(predictions))
        ]
        return jsonify({"confidences": confidences})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Mask Detector API is running!"

model = tf.keras.models.load_model("mask_detector.keras")
with open("class_mapping.json") as f:
    idx_to_class = json.load(f)

labels_fr = {
    "with_mask": "Masque détecté",
    "without_mask": "Sans masque",
    "mask_weared_incorrect": "Masque mal porté"
}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)