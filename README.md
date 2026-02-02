# indian-sign-language-recognition

<p align="center">
  <img src="https://cdn.prod.website-files.com/6023fdfa97944f09d6a27ac6/635b737d2d96d96840b33ed7_How%20to%20learn%20Basic%20Conversational%20Sign%20Language.webp" width="600">
</p>
This project implements a real-time Indian Sign Language (ISL) word recognition system using computer vision and machine learning.

## Features
- Real-time hand gesture recognition
- Supports common ISL word symbols
- Landmark-based classification using MediaPipe
- Neural network model built with TensorFlow
- Simple and clean UI using Streamlit

## Tech Stack
- Python
- OpenCV
- MediaPipe
- TensorFlow (Keras)
- Streamlit

## How It Works
1. Webcam captures hand gestures
2. MediaPipe extracts 21 hand landmarks
3. Landmarks are fed into a trained neural network
4. The system predicts the corresponding ISL word
5. Result is displayed in real time

## Project Structure
