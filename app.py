# -*- coding: utf-8 -*-
"""
Created on Tue May 13 23:29:30 2025

@author: HARSHIT NARAIN
"""

# streamlit_app.py
import streamlit as st
import tempfile
from predict import predict_image, predict_video

# --- Page Config ---
st.set_page_config(page_title="Deepfake Detector", layout="centered")

# --- Session State for Login ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

# --- Dummy User Store ---
users = {
    "admin@gmail.com": "admin123",
    "user@gmail.com": "user123"
}

# --- Background Styling ---
st.markdown("""
    <style>
    body {
        background: url('https://images.unsplash.com/photo-1535223289827-42f1e9919769?auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
        background-position: center;
    }
    .main {
        background-color: rgba(0, 0, 0, 0.6);
        padding: 3rem;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        color: white;
    }
    input, select, button {
        border-radius: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Login Form ---
def login():
    st.markdown("<h2 style='color:white;'>Login to Deepfake Detector</h2>", unsafe_allow_html=True)
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In")
        if submitted:
            if email in users and users[email] == password:
                st.session_state.authenticated = True
                st.session_state.user_email = email
                st.success("Login successful!")
            else:
                st.error("Invalid credentials")

# --- Signup Form ---
def signup():
    st.markdown("<h2 style='color:white;'>Create Account</h2>", unsafe_allow_html=True)
    with st.form("signup_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign Up")
        if submitted:
            if email in users:
                st.error("User already exists")
            else:
                users[email] = password
                st.success("Account created. Please log in.")

# --- Deepfake Detection Interface ---
def main_interface():
    st.markdown("<h2 style='color:white;'>Deepfake Detection</h2>", unsafe_allow_html=True)
    mode = st.radio("Select Mode", ["Image", "Video"])
    uploaded = st.file_uploader("Upload File", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())
        filepath = tfile.name

        if mode == "Image":
            st.image(filepath, caption="Uploaded Image", use_column_width=True)
            st.write("🔍 Detecting...")
            result = predict_image(filepath)
        else:
            st.video(filepath)
            st.write("🔍 Detecting...")
            result = predict_video(filepath)

        st.success(result)

# --- Authenticated User View ---
def logged_in_view():
    st.markdown(f"<h3 style='color:white;'>Welcome, {st.session_state.user_email}</h3>", unsafe_allow_html=True)
    main_interface()
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.user_email = ""
        st.experimental_rerun()

# --- Layout Switch ---
st.markdown("<div class='main'>", unsafe_allow_html=True)

menu = st.sidebar.radio("Navigation", ["Login", "Sign Up", "Try Without Login"])

if st.session_state.authenticated:
    logged_in_view()
elif menu == "Login":
    login()
elif menu == "Sign Up":
    signup()
elif menu == "Try Without Login":
    main_interface()

st.markdown("</div>", unsafe_allow_html=True)
