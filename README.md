# FingerCountModel

## Hand Gesture Finger Count Predictor

A real-time hand gesture recognition system that detects and counts fingers (0-5) using computer vision and deep learning. This project uses MediaPipe for hand landmark detection and a custom neural network for gesture classification.

## Features

- **Real-time Hand Detection**: Uses MediaPipe Hands to detect and track hand landmarks in video streams
- **Multi-class Classification**: Recognizes 6 different hand gestures (0-5 fingers)
- **High Accuracy**: Trained neural network achieves high accuracy in finger counting
- **User-friendly Interface**: Visual feedback with bounding boxes and prediction labels
- **Customizable Model**: Easily extendable architecture for additional gesture types

## Data Collection and Labeling

The data labeler script captures hand landmark data using MediaPipe and allows labeling with keyboard inputs:

- Press keys 0-5 to label the current hand pose with the corresponding finger count
- Each frame captures 21 hand landmarks (x, y, z coordinates) resulting in 63 features
- Data is saved to CSV with appropriate labels for model training

### Data Distribution

The dataset contains balanced samples of each gesture class (0-5 fingers) to ensure robust model training. The data collection process focused on capturing various hand orientations and positions to improve model generalization.

## Model Architecture

The SignModel is a deep neural network with multiple fully connected layers:

- Input layer: 63 features (21 landmarks × 3 coordinates)
- Hidden layers with dropout for regularization
- Output layer: 6 classes (0-5 fingers)

## Usage

### Prerequisites

- Python 3.7+
- PyTorch
- OpenCV
- MediaPipe
- NumPy
- scikit-learn

### Running the Application

1. Clone the repository
2. Install dependencies: `pip install torch opencv-python mediapipe numpy scikit-learn`
3. Run the main application: `python main.py`
