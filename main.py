from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Loading pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Loading preloaded image and convert it to grayscale
preloaded_image_path = 'preloaded_image.jpg'  
preloaded_image = cv2.imread(preloaded_image_path)
gray_preloaded_image = cv2.cvtColor(preloaded_image, cv2.COLOR_BGR2GRAY)

# Detecting face in the preloaded image
faces_preloaded = face_cascade.detectMultiScale(gray_preloaded_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

if len(faces_preloaded) == 0:
    raise Exception("No face detected in the preloaded image.")



cap = cv2.VideoCapture(0)
recognized = False

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return 

@app.route('/status')
def status():
    print(f"Current recognition status: {recognized}") 
    return jsonify({"recognized": recognized})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
    