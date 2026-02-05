# backend/app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from face_utils import decode_image, detect_faces
import pickle
import numpy as np
import os
import cv2
from sklearn.neighbors import KNeighborsClassifier

# --- CHANGE 1: Setup Flask to serve the React build folder ---
app = Flask(__name__, static_folder='build', static_url_path='')
CORS(app)

# --- LOAD THE TRAINED MODEL ---
data_folder = "data"
data_file = os.path.join(data_folder, "faces_data.pkl")
labels_file = os.path.join(data_folder, "labels.pkl")

knn = None

if os.path.exists(data_file) and os.path.exists(labels_file):
    print("Loading known faces...")
    with open(data_file, 'rb') as f:
        X = pickle.load(f)
    with open(labels_file, 'rb') as f:
        y = pickle.load(f)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    print("Model trained and ready!")
else:
    print("No training data found.")

# --- CHANGE 2: The Homepage Route ---
# When someone goes to your URL, give them the React app
@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

# ... (Keep imports and load model code same as before) ...

@app.route('/predict', methods=['POST'])
def predict():
    if not knn:
        return jsonify({"message": "Server not trained yet"}), 500

    try:
        data = request.get_json()
        img = decode_image(data['image'])
        
        # Detect Faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(img)

        if len(faces) == 0:
            return jsonify({"message": "No face detected", "face_count": 0})

        identified_names = []
        
        # --- NEW LOGIC: Loop through ALL faces ---
        for (x, y, w, h) in faces:
            # 1. Crop and Resize face
            face_section = gray[y:y+h, x:x+w]
            face_section = cv2.resize(face_section, (50, 50))
            face_vector = [face_section.flatten()]

            # 2. Get Prediction AND Probability (Confidence)
            # predict_proba returns the percentage of votes (e.g., [0.2, 0.8])
            probabilities = knn.predict_proba(face_vector)[0]
            max_prob = np.max(probabilities)
            prediction = knn.predict(face_vector)[0]

            # 3. Confidence Threshold
            # If less than 60% of neighbors agree, it's likely an unknown person or error
            if max_prob >= 0.6:
                identified_names.append(prediction)
            else:
                identified_names.append("Unknown")

        # --- Format the Final Message ---
        # Removes duplicates (so it doesn't say "Anurag, Anurag")
        unique_names = list(set(identified_names))
        
        if "Unknown" in unique_names and len(unique_names) == 1:
            message = "Face detected but not recognized."
        else:
            names_str = ", ".join([n for n in unique_names if n != "Unknown"])
            message = f"Hello, {names_str}!"
            if "Unknown" in unique_names:
                message += " (and Unknown person)"

        return jsonify({
            "message": message,
            "names": unique_names,
            "face_count": len(faces)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(port=5000, debug=True)