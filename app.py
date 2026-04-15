import os
import random
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes (to allow React to talk to Flask)

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Try to load the model, but if it fails, we fall back to mock predictions
MODEL_LOADED = False
try:
    import tensorflow as tf
    import numpy as np
    import cv2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    MODEL_PATH = "brain_tumor_model_fast.h5"
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        MODEL_LOADED = True
        print(f"Successfully loaded {MODEL_PATH}")
    else:
        print(f"Warning: {MODEL_PATH} not found. Running in MOCK mode.")
except Exception as e:
    print(f"Exception while loading model: {e}. Running in MOCK mode.")

CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
IMG_SIZE = 112

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Real Inference
        if MODEL_LOADED:
            try:
                img = cv2.imread(filepath)
                if img is None:
                    return jsonify({"error": "Invalid image format"}), 400
                
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = preprocess_input(img)
                img = np.expand_dims(img, axis=0)

                pred = model.predict(img, verbose=0)
                idx = np.argmax(pred)
                confidence = float(pred[0][idx])
                predicted_class = CLASS_NAMES[idx]

                return jsonify({
                    "className": predicted_class,
                    "confidence": confidence,
                    "mocked": False
                })
            except Exception as e:
                print("Error during inference:", e)
                return jsonify({"error": f"Inference error: {str(e)}"}), 500
            
        # Mock Inference (User doesn't have the model yet)
        else:
            time.sleep(1.5)  # Simulate processing delay
            # Mock some random result
            mock_class = random.choice(CLASS_NAMES)
            mock_confidence = random.uniform(0.70, 0.99)
            
            return jsonify({
                "className": mock_class,
                "confidence": mock_confidence,
                "mocked": True,
                "message": "NOTE: This is a simulated result because the model file was not found locally."
            })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
