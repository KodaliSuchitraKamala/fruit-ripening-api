from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load SavedModel using TFSMLayer
model = tf.keras.layers.TFSMLayer("model", call_endpoint="serving_default")
LABELS = ['Unripe', 'Ripe', 'Overripe']

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    image = Image.open(file.stream).convert("RGB").resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    preds = model(image)
    result = LABELS[np.argmax(preds)]
    return jsonify({'result': result})

@app.route("/")
def home():
    return "API working"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
