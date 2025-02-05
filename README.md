# Face Recognition & Analysis with DeepFace and TensorFlow

## Overview
This project implements a real-time face recognition and analysis system using TensorFlow for facial recognition and DeepFace for emotion, age, gender, and race detection. The system captures video from the webcam, detects faces, recognizes individuals, and performs facial analysis with DeepFace.

## Features
- Real-time face detection and recognition.
- Recognizes known faces from a set of pre-trained models.
- Analyzes face attributes such as age, gender, emotion, and race using DeepFace.
- Displays the recognition and analysis results on the video feed.
- Optimized performance by limiting DeepFace processing to every 10 frames.

## Prerequisites
Before running this project, ensure you have the following:
- Python 3 installed.
- Required Python packages installed:
  - `tensorflow`
  - `deepface`
  - `opencv-python`
  - `numpy`
  
## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the following files:
   - **Pre-trained face recognition model** (`Face_Recognition_Model.h5`): Ensure you have your trained model in the specified directory or adjust the path in the script.
   - **Haar Cascade XML file** (`harr_face.xml`): This is used for face detection. Make sure it's placed in the correct directory or adjust the path.

4. Ensure you have a working webcam connected to your system for real-time face detection.

## Usage
1. Modify the paths to your pre-trained model and Haar Cascade XML file in the script:
   - `Face_Recognition_Model.h5`: Update this to the correct location of your model file.
   - `harr_face.xml`: Update this to the correct location of the Haar Cascade XML file for face detection.

2. Run the script:
   ```bash
   python src/main.py
   ```

   The system will open a video window displaying the webcam feed, where it will perform face detection and recognition in real-time. The recognized face will be labeled with the person's name, and additional information like age, gender, emotion, and race will be shown using DeepFace.

3. Press `q` to quit the video feed.

## File Structure
```
📂 Project Root
├── 📂 src
│   ├── main.py  # Main script to capture video, perform face recognition & analysis
│
├── 📂 models
│   └── Face_Recognition_Model.h5  # Your pre-trained face recognition model
│
├── 📂 haar_cascade
│   └── harr_face.xml  # Haar cascade file for face detection
│
├── requirements.txt  # List of dependencies
├── README.md  # Project documentation
```

## Key Components
1. **Face Recognition**: Uses a TensorFlow pre-trained model to recognize faces from the webcam feed. The model is loaded from the `.h5` file and predicts faces based on predefined class labels.
   
2. **Face Detection**: Uses Haar Cascade to detect faces in the video frames, which are then passed to the recognition model.

3. **DeepFace Analysis**: Every 10 frames, the system analyzes the detected face with DeepFace to get insights such as:
   - Age
   - Gender
   - Emotion
   - Race

   The results are displayed along with the face recognition label.

## API Quota Considerations
- DeepFace performs facial analysis using external models, so ensure you have a stable internet connection if you're using remote analysis.

## Future Enhancements
- Improve face recognition accuracy by retraining the model with more faces.
- Integrate the system with a database to track and store recognized faces.
- Implement real-time notifications for specific face detections (e.g., security alerts).
  
