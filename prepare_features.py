# -*- coding: utf-8 -*-
"""
Created on Wed May 14 00:40:40 2025

@author: HARSHIT NARAIN
"""

import os
import numpy as np
from utils.feature_extractor import extract_features
from tensorflow.keras.preprocessing import image

face_dir = "data/faces"
video_features = []
video_labels = []

# Group images by video ID
videos = {}
for fname in os.listdir(face_dir):
    if fname.endswith('.jpg'):
        video_id = '_'.join(fname.split('_')[:1])  # only first part (e.g., 183)
        videos.setdefault(video_id, []).append(os.path.join(face_dir, fname))

for vid, paths in videos.items():
    paths = sorted(paths)[:10]  # limit to first 10 frames
    if len(paths) < 5:
        print(f"Skipping {vid}: only {len(paths)} frames")
        continue

    # Load and resize images
    faces = [image.load_img(p, target_size=(224, 224)) for p in paths]
    features = extract_features(faces)  # (<=10, 2048)

    # Pad with zeros if less than 10
    if features.shape[0] < 10:
        pad_len = 10 - features.shape[0]
        padding = np.zeros((pad_len, 2048))
        features = np.vstack([features, padding])

    video_features.append(features)

    # Label: 1 = fake, 0 = real
    label = 1 if 'fake_' in vid else 0
    video_labels.append(label)

# Save dataset
X = np.array(video_features)
y = np.array(video_labels)
np.save('data/X.npy', X)
np.save('data/y.npy', y)

print(f"✅ Saved X.npy with shape {X.shape}")
print(f"✅ Saved y.npy with shape {y.shape}")
