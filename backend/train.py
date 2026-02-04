# backend/train.py
import cv2
import numpy as np
import os
import pickle

# 1. Setup Data Folder
data_folder = "data"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# 2. Ask for the name
name = input("Enter your name: ")
print(f"Get ready! We will take 20 photos of {name}.")
print("Press 'c' to capture a photo. Press 'q' to quit.")

cap = cv2.VideoCapture(0) # Open Webcam
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_data = [] # To store the face image data
labels = []    # To store the name

count = 0
MAX_FACES = 20

while True:
    ret, frame = cap.read()
    if not ret: continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw a box around the face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Trainer", frame)

    # Key Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'): # If 'c' is pressed
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            
            # Crop and Resize the face to 50x50 pixels
            face_section = gray[y:y+h, x:x+w]
            face_section = cv2.resize(face_section, (50, 50))
            
            # Save the face data
            face_data.append(face_section.flatten()) 
            labels.append(name)
            
            count += 1
            print(f"Captured {count}/{MAX_FACES}")
        else:
            print("No face found! Look at the camera.")

    if count >= MAX_FACES:
        break
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 3. Save the Data
if count > 0:
    # Save faces and names to files
    with open(os.path.join(data_folder, "faces_data.pkl"), 'wb') as f:
        pickle.dump(face_data, f)
    with open(os.path.join(data_folder, "labels.pkl"), 'wb') as f:
        pickle.dump(labels, f)
    print("Training Complete! Data saved.")
else:
    print("No data saved.")