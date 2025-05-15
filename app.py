# -*- coding: utf-8 -*-
"""
Created on Tue May 13 23:29:30 2025

@author: HARSHIT NARAIN
"""

import streamlit as st
import tempfile
from predict import predict_video, predict_image_with_cnn

st.set_page_config(page_title="Deepfake Detector")
st.title("🧠 Deepfake Detection")
mode = st.radio("Select input type:", ["Image", "Video"])

uploaded_file = st.file_uploader("Upload file", type=["mp4", "jpg", "jpeg", "png"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    if mode == "Video":
        st.video(tfile.name)
        st.write("🔍 Detecting deepfake in video...")
        result = predict_video(tfile.name)
    else:
        st.image(tfile.name)
        st.write("🔍 Detecting deepfake in image...")
        result = predict_image_with_cnn(tfile.name)

    st.success(result)
