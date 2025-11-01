# Try-On Filter Application - Copilot Instructions

## Project Overview
Desktop AR filter application using classical ML for face tracking (no OpenCV/MediaPipe) with Godot for visualization.

## Technology Stack
- **Python**: Core ML training and real-time tracking
- **scikit-learn**: KNN/Naive Bayes/Decision Tree classifiers
- **NumPy**: Manual image processing and HSV conversion
- **Pillow**: Image I/O without OpenCV
- **Godot Engine**: AR overlay and webcam display
- **Socket/UDP**: Python-Godot communication

## Key Constraints
- NO OpenCV, MediaPipe, dlib, or deep learning models
- Face tracking MUST use self-trained classical ML
- All image processing implemented manually
- HSV conversion coded from scratch
- Connected component analysis implemented manually

## Project Structure
- `python_ml_tracking/`: ML training and tracking modules
- `godot_project/`: Godot scenes and scripts
- `datasets/`: Training images and labels
- `models/`: Trained ML models (pickle files)
- `docs/`: Architecture and implementation docs
