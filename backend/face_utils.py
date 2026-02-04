# backend/face_utils.py
import cv2
import numpy as np
import base64

def decode_image(base64_string):
    """
    This function takes the text string (Base64) from React 
    and converts it back into an image (NumPy array) that OpenCV can use.
    """
    # 1. Clean up the string (remove the "data:image/jpeg;base64," part if present)
    if ',' in base64_string:
        header, encoded = base64_string.split(",", 1)
    else:
        encoded = base64_string

    # 2. Decode the base64 string into raw bytes
    image_bytes = base64.b64decode(encoded)

    # 3. Convert bytes into a numpy array of numbers
    nparr = np.frombuffer(image_bytes, np.uint8)

    # 4. Decode the numpy array into an OpenCV image (BGR color format)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return img

def detect_faces(img):
    """
    This function looks at the image and returns a list of faces found.
    Each face is a list of 4 numbers: [x, y, width, height].
    """
    # Load the pre-trained face detector file (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to Grayscale (black & white) because it's faster for AI
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    # scaleFactor=1.1: reduce image size by 10% each pass to find big and small faces
    # minNeighbors=4: higher number means fewer false positives (more strict)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    
    return faces