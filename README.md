# sign-langauge-translator-for-ISL
 
<p align="center">
 <p align="center">
 
</p>

</p>![635b737d2d96d96840b33ed7_How to learn Basic Conversational Sign Language](https://github.com/user-attachments/assets/8b1d280c-87dc-4c6f-a6ae-911f0662bb67)


<h1 align="center">Indian Sign Language Gesture Translator</h1>

<p align="center">
  A real-time Indian Sign Language (ISL) gesture recognition system using hand landmarks, temporal modeling, and deep learning.
</p>

---

## 📌 Overview

This project implements a **real-time Indian Sign Language (ISL) gesture recognition system** using computer vision and machine learning.  
Instead of relying on raw image classification, the system uses **hand landmark-based feature extraction** and **temporal sequence modeling** to achieve robust and interpretable gesture recognition.

The goal of this project is to build a **lightweight, explainable, and extensible ISL recognition pipeline** suitable for academic projects and real-time applications.

---

## 🚀 Key Features

- Real-time hand detection using MediaPipe  
- Landmark stabilization (position normalization, scale normalization, temporal smoothing)  
- Geometric feature extraction (inter-finger distances and joint angles)  
- Temporal modeling using sliding window sequences  
- Support for **single-hand and two-hand gestures**  
- Dataset recording for custom gesture training  
- Web-based UI using **HTML + Flask**  
- Modular and extensible pipeline design  

---

## 🧠 Methodology

The system follows a structured pipeline:

