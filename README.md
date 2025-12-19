# indian-sign-language-recognition

![System Architecture](https://www.google.com/url?sa=t&source=web&rct=j&url=https%3A%2F%2Fwww.indiancentury.in%2F2022%2F09%2F26%2Fincreasing-awareness-about-indian-sign-language%2F&opi=89978449)

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
