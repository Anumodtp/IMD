from flask import Flask, request, render_template, redirect, url_for, flash, send_file, Response
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
from io import BytesIO
import io

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Configure the upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Load the trained deepfake image manipulation detection model
model = load_model('model_xception_deepfake.h5')

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define a function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

            # Convert the image to grayscale for face detection
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Check if any faces were detected
            if len(faces) == 0:
                print("No faces detected in the image.")
            else:
                # Initialize a dictionary to store unique faces and their predictions
                face_data = {}

                # Iterate through the detected faces
                for (x, y, w, h) in faces:
                    # Crop the image around the detected face
                    face_crop = image[y:y + h, x:x + w]

                    # Resize the cropped face to match the model's input shape (224, 224)
                    face_crop_resized = cv2.resize(face_crop, (224, 224))

                    # Normalize the face crop
                    face_crop_normalized = face_crop_resized / 255.0

                    # Predict using the model
                    prediction = model.predict(np.expand_dims(face_crop_normalized, axis=0))[0][0]

                    # Store the unique face and its prediction in the dictionary
                    face_data[(x, y, w, h)] = prediction

                # Define a threshold value for classification
                threshold = 0.5

                # Clear any previous plot
                plt.clf()

                # Create a new plot for the current image
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # Iterate through unique face data and plot results
                for (x, y, w, h), prediction in face_data.items():
                    # Determine whether the face is real or fake based on the prediction
                    label = "Fake" if prediction > threshold else "Real"
                    color = 'blue' if label == "Fake" else 'green'

                    # Plot bounding box and label
                    plt.gca().add_patch(plt.Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none'))
                    plt.text(x, y - 10, label, color='white', backgroundcolor=color)

                # Save the plot image to an in-memory file
                img_io = io.BytesIO()
                plt.axis('off')
                plt.savefig(img_io, format='png')

                # Reset the buffer's position to the beginning
                img_io.seek(0)

                # Return the in-memory file as a response
                return Response(img_io.getvalue(), content_type='image/png')

    # Render the upload page
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
