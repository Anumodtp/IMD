from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from mtcnn import MTCNN
import os
import base64
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model  # Add this import
import numpy as np  # Add this import

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Configure the upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Initialize MTCNN for face detection
detector = MTCNN()

# Define padding percentage
padding_percentage = 0.11

# Define a function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess a single image and predict its class
def predict_single_image(image_path):
    model = load_model('updated_model.h5')
    img = load_img(image_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    print(prediction)
    if prediction[0][0] < 0.5:
        return "Fake"
    else:
        return "Real"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        # Check if the user has selected a file
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        # Check if the file has an allowed extension
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the uploaded file
            file.save(file_path)

            # Load the image using OpenCV
            image = cv2.imread(file_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect faces in the image
            faces = detector.detect_faces(image_rgb)

            # Check if any faces were detected
            if len(faces) == 0:
                flash("No faces detected in the uploaded image.")
                return redirect(request.url)
            else:
                # Iterate through the detected faces
                for face in faces:
                    x, y, w, h = face['box']
                    # Calculate padding
                    pad_x = int(w * padding_percentage)
                    pad_y = int(h * padding_percentage)
                    # Expand bounding box coordinates
                    x -= pad_x
                    y -= pad_y
                    w += 2 * pad_x
                    h += 2 * pad_y
                    # Ensure the bounding box is within image bounds
                    x = max(x, 0)
                    y = max(y, 0)
                    w = min(w, image.shape[1] - x)
                    h = min(h, image.shape[0] - y)

                    # Crop the image around the expanded bounding box
                    padded_face = image_rgb[y:y+h, x:x+w]

                    # Preprocess the padded face
                    preprocessed_face = preprocess_input(cv2.resize(padded_face, (299, 299)))

                    # Make predictions using the model
                    prediction_label = predict_single_image(file_path)

                    # Determine border color based on prediction
                    border_color = (0, 255, 0) if prediction_label == "Real" else (255, 0, 0)

                    # Draw bounding box around the detected face with border color
                    cv2.rectangle(image_rgb, (x, y), (x + w, y + h), border_color, 2)

                    # Put prediction label on top of bounding box
                    cv2.putText(image_rgb, prediction_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, border_color, 2)

                # Encode the image to base64 for embedding in HTML
                _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
                img_data = base64.b64encode(img_encoded).decode('utf-8')

                # Render the result page with the image data
                return render_template('index.html', img_data=img_data)

    # Render the upload page
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
