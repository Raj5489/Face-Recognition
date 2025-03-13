# Face Recognition System

A real-time face recognition system using Python, MediaPipe, and OpenCV. This system can detect faces, draw facial landmarks, and recognize known faces from your webcam feed.

## Features

- Real-time face detection
- Facial landmark detection
- Face recognition
- Add new faces to the system
- Interactive menu interface

## Requirements

- Python 3.7 or higher
- Webcam
- Required packages (install using `pip install -r requirements.txt`):
  - mediapipe
  - opencv-python
  - numpy
  - face-recognition
  - dlib

## Installation

1. Clone this repository or download the files
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the program:
```bash
python face_recognition_system.py
```

2. Choose from the menu options:
   - Option 1: Start Face Recognition
   - Option 2: Add New Face
   - Option 3: Exit

### Adding New Faces

1. Select option 2 from the menu
2. Enter the person's name
3. Look at the camera and press 'c' to capture your face
4. The image will be saved in the 'known_faces' directory

### Running Face Recognition

1. Select option 1 from the menu
2. The system will start your webcam and begin detecting and recognizing faces
3. Press 'q' to quit the face recognition mode

## Directory Structure

```
face_recognition_system/
├── face_recognition_system.py
├── requirements.txt
├── README.md
└── known_faces/
    └── (your face images will be stored here)
```

## Controls

- Press 'q' to quit the face recognition or face capture mode
- Press 'c' to capture a face when adding a new person

## Notes

- Make sure you have good lighting when adding new faces or using face recognition
- The system works best when faces are clearly visible and facing the camera
- Multiple faces can be detected, but only one face will be processed for landmarks 