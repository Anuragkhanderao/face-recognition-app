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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # The important part: this returns a LIST of rectangles, not just one.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    return faces