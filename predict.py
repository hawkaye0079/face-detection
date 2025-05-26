import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from utils.preprocess import extract_frames, detect_faces
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model_cnn = load_model('model/image_model_augmented.h5')
model_lstm = load_model('model/lstm_model_retrained.keras')

# Reuse ResNet for both video prediction and feature extraction
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def predict_image(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))
        x = img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        pred = model_cnn.predict(x)[0][0]
        label = "🔴 Deepfake" if pred >= 0.5 else "🟢 Real"
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

        processed_faces = []
        for face in faces:
            face = cv2.resize(face, (224, 224))
            face = preprocess_input(face.astype('float32'))
            processed_faces.append(face)

        processed_faces = np.array(processed_faces)
        features = resnet_model.predict(processed_faces, verbose=0)  
        features = np.expand_dims(features, axis=0)  

        pred = model_lstm.predict(features)[0][0]
        label = "🔴 Deepfake" if pred >= 0.5 else "🟢 Real"
        return f"{label} ({pred*100:.2f}% confidence)"
    except Exception as e:
        return f"Error: {str(e)}"

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
    pred = model_cnn.predict(array)[0][0]
    label = "🔴 Deepfake" if pred >= 0.5 else "🟢 Real"
    return f"Prediction: {label} ({pred*100:.2f}% confidence)"
