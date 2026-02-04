# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from face_utils import decode_image, detect_faces
import pickle
import numpy as np
import os
import cv2
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
CORS(app)

# --- LOAD THE TRAINED MODEL ---
data_folder = "data"
data_file = os.path.join(data_folder, "faces_data.pkl")
labels_file = os.path.join(data_folder, "labels.pkl")

knn = None

# Check if data exists
if os.path.exists(data_file) and os.path.exists(labels_file):
    print("Loading known faces...")
    with open(data_file, 'rb') as f:
        X = pickle.load(f) # Face Data
    with open(labels_file, 'rb') as f:
        y = pickle.load(f) # Names
    
    # Train the K-NN Classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    print("Model trained and ready!")
else:
    print("No training data found. Run train.py first!")

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "online"})

@app.route('/predict', methods=['POST'])
def predict():
    if not knn:
        return jsonify({"message": "Server not trained yet"}), 500

    try:
        data = request.get_json()
        img = decode_image(data['image'])
        
        # 1. Detect Face
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(img)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            
            # 2. Process Face for Recognition
            face_section = gray[y:y+h, x:x+w]
            face_section = cv2.resize(face_section, (50, 50))
            face_vector = [face_section.flatten()]

            # 3. Predict Name
            prediction = knn.predict(face_vector)
            name = prediction[0]

            return jsonify({
                "message": f"Hello, {name}!",
                "name": name,
                "face_count": 1
            })
        else:
            return jsonify({"message": "No face detected", "face_count": 0})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)