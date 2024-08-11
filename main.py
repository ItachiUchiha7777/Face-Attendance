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

(x, y, w, h) = faces_preloaded[0]
preloaded_face = gray_preloaded_image[y:y+w, x:x+h]

cap = cv2.VideoCapture(0)
recognized = False

def gen_frames():
    global recognized  # Declare global to modify the variable
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        recognized = False  # Reset at the beginning of each frame processing

        for (x, y, w, h) in faces:
            face = gray_frame[y:y+w, x:x+h]
            resized_face = cv2.resize(face, (preloaded_face.shape[1], preloaded_face.shape[0]))
            score = cv2.matchTemplate(resized_face, preloaded_face, cv2.TM_CCOEFF_NORMED)
            similarity = np.max(score)
            
            if similarity > 0.4:
                recognized = True
                color = (0, 255, 0)  # Green
                text = "Face Matched"
            else:
                color = (0, 0, 255)  # Red
                text = "Face Not Matched"
                
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Encode the frame in JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame to the client
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    print(f"Current recognition status: {recognized}") 
    return jsonify({"recognized": recognized})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
    