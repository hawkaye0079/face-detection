# -*- coding: utf-8 -*-
"""
Created on Tue May 13 23:29:19 2025

@author: HARSHIT NARAIN
"""

from tensorflow.keras.models import load_model
from utils.preprocess import extract_frames, detect_faces
from utils.feature_extractor import extract_features
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Load model once
model = load_model('model/lstm_model.h5')

def predict_video(video_path):
    frames = extract_frames(video_path, None)
    faces = detect_faces(frames)
    if len(faces) < 5:
        return "❗ Not enough face data for prediction."

    faces = faces[:10]
    features = extract_features(faces)
    features = np.expand_dims(features, axis=0)
    pred = model.predict(features)[0][0]
    label = "🟢 Real" if pred < 0.5 else "🔴 Deepfake"
    return f"Prediction: {label} ({pred*100:.2f}% confidence)"

def predict_image(image_path):
    img = cv2.imread(image_path)
    faces = detect_faces([img])
    if not faces:
        return "❗ No face detected in the image."

    face = faces[0]
    features = extract_features([face])
    features = np.expand_dims(features, axis=0)
    features = np.pad(features, ((0, 9), (0, 0)))  # pad to 10 frames
    features = np.expand_dims(features, axis=0)

    pred = model.predict(features)[0][0]
    label = "🟢 Real" if pred < 0.5 else "🔴 Deepfake"
    return f"Prediction: {label} ({pred*100:.2f}% confidence)"

# Load CNN image classifier
cnn_model = load_model('model/image_model.h5')

def draw_face_box(image_path):
    from mtcnn import MTCNN
    img = cv2.imread(image_path)
    detector = MTCNN()
    result = detector.detect_faces(img)
    if result:
        x, y, w, h = result[0]['box']
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def predict_image_with_cnn(image_path):
    image = load_img(image_path, target_size=(224, 224))
    array = img_to_array(image)
    array = np.expand_dims(array, axis=0) / 255.0
    pred = cnn_model.predict(array)[0][0]
    label = "🟢 Real" if pred < 0.5 else "🔴 Deepfake"
    return f"Prediction: {label} ({pred*100:.2f}% confidence)"