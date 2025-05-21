# -*- coding: utf-8 -*-
"""
Created on Tue May 13 23:29:19 2025

@author: HARSHIT NARAIN
"""

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from utils.preprocess import extract_frames, detect_faces
from utils.feature_extractor import extract_features
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model_cnn = load_model('model/image_model.h5')
model_lstm = load_model('model/lstm_model.h5')

def predict_image(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))
        x = img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        pred = model_cnn.predict(x)[0][0]
        label = "🟢 Real" if pred > 0.5 else "🔴 Deepfake"
        return f"{label} ({pred*100:.2f}% confidence)"
    except Exception as e:
        return f"Error: {str(e)}"

def predict_video(video_path):
    try:
        frames = extract_frames(video_path, None, max_frames=30)
        faces = detect_faces(frames)

        if len(faces) < 10:
            return "❗ Not enough detectable faces in video."

        faces = faces[:10]
        features = extract_features(faces)
        features = np.expand_dims(features, axis=0)

        pred = model_lstm.predict(features)[0][0]
        label = "🟢 Real" if pred >= 0.5 else "🔴 Deepfake"
        return f"{label} ({pred*100:.2f}% confidence)"
    except Exception as e:
        return f"Error: {str(e)}"


# Load CNN image classifier
cnn_model = load_model('model/image_model.h5')

def predict_image(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))
        x = img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        pred = model_cnn.predict(x)[0][0]
        label = "🟢 Real" if pred >= 0.5 else "🔴 Deepfake"
        return f"{label} ({pred*100:.2f}% confidence)"
    except Exception as e:
        return f"Error: {str(e)}"

def predict_image_with_cnn(image_path):
    try:
        image = load_img(image_path, target_size=(224, 224))
        array = img_to_array(image) / 255.0
        array = np.expand_dims(array, axis=0)
        pred = cnn_model.predict(array)[0][0]
        label = "🟢 Real" if pred >= 0.5 else "🔴 Deepfake"
        return f"Prediction: {label} ({pred*100:.2f}% confidence)"
    except Exception as e:
        return f"Error: {str(e)}"
