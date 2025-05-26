from flask import Flask, request, jsonify, send_from_directory, render_template_string
import os
from werkzeug.utils import secure_filename
from predict import predict_image, predict_video
from PIL import Image
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the frontend HTML
with open("html_frontend.html", "r", encoding="utf-8") as f:
    FRONTEND_HTML = f.read()

@app.route('/')
def home():
    return render_template_string(FRONTEND_HTML)

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': True})

@app.route('/api/detect', methods=['POST'])
def detect_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    filetype = file.content_type  # e.g. image/png or video/mp4

    try:
        if filetype.startswith('image'):
            result_text = predict_image(filepath)
        elif filetype.startswith('video'):
            result_text = predict_video(filepath)
        else:
            return jsonify({'success': False, 'error': 'Unsupported file type'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Prediction error: {str(e)}'})

    # Parse result text like: "🟢 Real (87.52% confidence)"
    try:
        label = "Real" if "🟢" in result_text else "Deepfake"
        confidence = float(result_text.split('(')[1].split('%')[0]) / 100
    except:
        label = "Unknown"
        confidence = 0.0

    # Calculate file size and dimensions
    file_size = os.path.getsize(filepath)
    image_dims = "-"
    if filetype.startswith("image"):
        try:
            with Image.open(filepath) as img:
                image_dims = f"{img.width}x{img.height}"
        except:
            image_dims = "?x?"
    elif filetype.startswith("video"):
        cap = cv2.VideoCapture(filepath)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            image_dims = f"{w}x{h}"
        cap.release()

    return jsonify({
        'success': True,
        'label': label,
        'confidence': confidence,
        'confidence_percentage': f"{confidence * 100:.2f}%",
        'real_probability': confidence if label == 'Real' else 1 - confidence,
        'fake_probability': 1 - confidence if label == 'Real' else confidence,
        'file_size': file_size,
        'image_dimensions': image_dims,
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
