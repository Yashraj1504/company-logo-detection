import os
import cv2
import numpy as np
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

# Initialize Flask App
app = Flask(__name__)
app.secret_key = "secret_key"

# Upload Folder Configuration
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

# Ensure Upload Directory Exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Model
MODEL_PATH = 'company_logo_1.h5'
model = load_model(MODEL_PATH)

# Dictionary Mapping for Predictions
company_labels = {
    0: 'AFFORDGEN PHARMACEUTICALS PRIVATE LIMITED',
    1: 'ANTON GLOBAL PHARMACEUTICALS PRIVATE LIMITED',
    2: 'ARKSA PHARMA PRIVATE LIMITED',
    3: 'AVIREX PHARMA PRIVATE LIMITED',
    4: 'AYONIX PHARMACEUTICAL PRIVATE LIMITED',
    5: 'BM HEALTH MART PHARMA PRIVATE LIMITED',
    6: 'HEALTHNOVO PHARMACEUTICAL PRIVATE LIMITED',
    7: 'INTERLOOP PHARMACEUTICALS PRIVATE LIMITED',
    8: 'JAKS PHARMA PRIVATE LIMITED',
    9: 'MEDVAC PHARMA PRIVATE LIMITED',
    10: 'METAMED PHARMA PRIVATE LIMITED',
    11: 'MUNANI PHARMACEUTICALS PRIVATE LIMITED',
    12: 'NAKSHATRA PHARMA IMPEX INDIA PRIVATE LIMITED',
    13: 'NUREK PHARMACEUTICALS PRIVATE LIMITED',
    14: 'OCURE PHARMA PRIVATE LIMITED',
    15: 'OPES PHARMA PRIVATE LIMITED',
    16: 'RVJ PHARMA CONSULTANTS PRIVATE LIMITED',
    17: 'S.A. PHARMACHEM (ANIMAL HEALTH) PRIVATE LIMITED',
    18: 'TAUSCHEN ZENVISION PHARMA INDI',
    19: 'TITAN PHARMAPLUS PRIVATE LIMITED',
    20: 'TRIDOT PHARMACEUTICAL PRIVATE LIMITED',
    21: 'VG ROYAL PHARMACY LIMITED',
    22: 'WINISTA PHARMACEUTICAL PRIVATE LIMITED'
}

# Function to Check Allowed File Type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to Predict Company from Logo Image
def predict_company(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0  
    image = np.expand_dims(image, axis=0)  # Reshape for model input

    prediction = model.predict(image)[0]  # Extract prediction result
    prediction_index = np.argmax(prediction)

    return company_labels.get(prediction_index, "Unknown Company")

# Route for Upload Form
@app.route('/')
def upload_form():
    return render_template('upload.html')

# Route for Handling File Upload & Prediction
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Predict Company
        result = predict_company(file_path)
        flash(f'Predicted Company: {result}')

        return render_template('upload.html', filename=filename)

    else:
        flash('Allowed file types: png, jpg, jpeg, gif')
        return redirect(request.url)

# Route to Display Uploaded Image
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename))

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
