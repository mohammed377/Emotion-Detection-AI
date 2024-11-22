import cv2
import numpy as np
from keras.models import load_model
import json

# Load the trained model
model = load_model('emotion_model.h5')

# Load the class labels
with open('class_labels.json', 'r') as json_file:
    class_labels = json.load(json_file)

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Real-Time Emotion Detection
def real_time_emotion_detection():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Preprocess the face
            roi = gray_frame[y:y+h, x:x+w]
            resized = cv2.resize(roi, (48, 48))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 48, 48, 1))

            # Predict the emotion
            prediction = model.predict(reshaped, verbose=0)
            emotion_label = class_labels[str(np.argmax(prediction))]

            # Draw rectangle and emotion label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show the frame
        cv2.imshow('Emotion Detection', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the detection
real_time_emotion_detection()
