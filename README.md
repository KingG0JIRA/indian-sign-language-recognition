# sign-langauge-translator-for-ISL
 
<p align="center">
 <img src="635b737d2d96d96840b33ed7_How to learn Basic Conversational Sign Language.webp"/>

 
 
</p>




<h1 align="center">Indian Sign Language Gesture Translator</h1>

<p align="center">
  A real-time Indian Sign Language (ISL) gesture recognition system using hand landmarks, temporal modeling, and deep learning.
</p>

---

##  Overview

This project implements a **real-time Indian Sign Language (ISL) gesture recognition system** using computer vision and machine learning.  
Instead of relying on raw image classification, the system uses **hand landmark-based feature extraction** and **temporal sequence modeling** to achieve robust and interpretable gesture recognition.

The goal of this project is to build a **lightweight, explainable, and extensible ISL recognition pipeline** suitable for academic projects and real-time applications.

---

##  Key Features

- Real-time hand detection using MediaPipe  
- Landmark stabilization (position normalization, scale normalization, temporal smoothing)  
- Geometric feature extraction (inter-finger distances and joint angles)  
- Temporal modeling using sliding window sequences  
- Support for **single-hand and two-hand gestures**  
- Dataset recording for custom gesture training  
- Web-based UI using **HTML + Flask**  
- Modular and extensible pipeline design  

---

## Methodology

The system follows a structured pipeline:
Camera Input
→ Hand Detection
→ Landmark Stabilization
→ Feature Extraction
→ Temporal Frame Buffer (30 frames)
→ Dataset Recording / Model Prediction


Each gesture is represented as a **(30 × 32)** feature sequence:
- **30** time steps (frames)
- **32** features per frame (16 per hand)

This approach captures **both hand shape and motion dynamics**, making it more robust than static image-based methods.

---

##  Technologies Used

- **Python**
- **OpenCV**
- **MediaPipe (Hand Landmarks)**
- **NumPy**
- **Flask** (Web UI)
- **TensorFlow / Keras** (LSTM model – training stage)

---

## Project Structure



isl-gesture-translator/
│
├── pipeline.py # Shared preprocessing & feature extraction
├── record_dataset.py # Dataset recording script
├── app.py # Flask backend
│
├── templates/
│ └── index.html # Web UI
│
├── static/
│ └── style.css # UI styling
│
├── dataset/ # Recorded gesture data (.npy)
│
├── model/ # Trained LSTM models
│
└── README.md


---

## Dataset

- Each gesture sample is stored as a **NumPy array (.npy)** of shape `(30, 32)`
- 20–30 samples are recorded per gesture
- Dataset is collected in a controlled environment
- The same preprocessing pipeline is used for **recording, training, and inference**

---

## Supported Gestures

This project focuses on a **limited subset of ISL gestures** for demonstration purposes:

- HELLO  
- YES  
- NO  
- THANK YOU  
- STOP  

The system can be extended easily by recording additional gestures.

---

## How to Run

### 1️⃣ Install Dependencies
```bash
pip install opencv-python mediapipe numpy flask tensorflow
