import cv2 as cv
import numpy as np
import tensorflow as tf
from deepface import DeepFace

# Load your trained model
model = tf.keras.models.load_model(r'model\Face_Recognition_Model.h5')

# Define class labels for your face recognition model
characters = ['Subject1', 'Subject2', 'Subject3', 'Unknown']  # Add more names if needed

# Load Haar Cascade for face detection
haar_cascade = cv.CascadeClassifier(r'haar_cascade\harr_face.xml')

# Function to prepare the image for TensorFlow model
def prepare(img):
    img_size = (80, 80)  # Ensure this matches the input size of your model
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv.resize(img, img_size)  # Resize
    img = img.reshape(-1, 80, 80, 1)  # Reshape for the model
    img = img / 255.0  # Normalize
    return img

# Start webcam capture
cap = cv.VideoCapture(0)

# Set the video capture to 60fps if possible (adjust the camera capture FPS)
cap.set(cv.CAP_PROP_FPS, 60)

frame_count = 0  # To limit DeepFace processing frequency

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture image.")
        break

    frame = cv.flip(frame, 1)  # Flip for mirror effect
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=7)

    for (x, y, w, h) in faces_rect:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract face region
        face_roi = frame[y:y + h, x:x + w]

        # Recognize face using TensorFlow model
        img_prepared = prepare(face_roi)
        predictions = model.predict(img_prepared)
        predicted_class_idx = np.argmax(predictions, axis=1)
        predicted_label = characters[predicted_class_idx[0]]

        # Display recognized name
        cv.putText(frame, predicted_label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Run DeepFace analysis every 10 frames to optimize performance
        if frame_count % 10 == 0:
            try:
                # Convert the face ROI to RGB and process with DeepFace
                face_rgb = cv.cvtColor(face_roi, cv.COLOR_BGR2RGB)  # DeepFace expects RGB
                analysis = DeepFace.analyze(img_path=face_rgb, actions=["age", "gender", "emotion", "race"], enforce_detection=False)

                # Extract results
                age = analysis[0]['age']
                gender = analysis[0]['dominant_gender']
                emotion = analysis[0]['dominant_emotion']
                race = analysis[0]['dominant_race']

                # Display DeepFace results
                info = f"{gender}, {age} yrs, {emotion}, {race}"
                cv.putText(frame, info, (x, y + h + 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            except Exception as e:
                print("DeepFace Error:", e)

    # Show video frame
    cv.imshow('Face Recognition & Analysis', frame)

    frame_count += 1

    # Break loop on 'q' key press
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release resources
cap.release()
cv.destroyAllWindows()
