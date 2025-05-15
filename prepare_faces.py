# -*- coding: utf-8 -*-
"""
Created on Wed May 14 00:40:24 2025

@author: HARSHIT NARAIN
"""

import os
import cv2
from utils.preprocess import extract_frames, detect_faces

input_dir = "data/original_sequences/youtube/raw/videos"
output_dir = "data/faces"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith('.mp4'):
        video_path = os.path.join(input_dir, filename)
        video_name = filename.split('.')[0]
        print(f"Processing: {video_name}")

        frames = extract_frames(video_path, None, max_frames=30)
        faces = detect_faces(frames)

        # Save faces as images
        for i, face in enumerate(faces):
            out_path = os.path.join(output_dir, f"{video_name}_{i}.jpg")
            cv2.imwrite(out_path, face)
# Add this at the bottom to handle both real and fake
fake_input_dir = "data/manipulated_sequences/Face2Face/raw/videos"
for filename in os.listdir(fake_input_dir):
    if filename.endswith('.mp4'):
        video_path = os.path.join(fake_input_dir, filename)
        video_name = "fake_" + filename.split('.')[0]
        print(f"Processing: {video_name}")

        frames = extract_frames(video_path, None, max_frames=30)
        faces = detect_faces(frames)

        for i, face in enumerate(faces):
            out_path = os.path.join(output_dir, f"{video_name}_{i}.jpg")
            cv2.imwrite(out_path, face)
