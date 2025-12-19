# indian-sign-language-recognition

<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRS6aNAkiYFcq2cz7LlnoI1q2v4kZW3wz4O-Q&s" width="600">
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
