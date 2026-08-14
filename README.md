# 🤟 Real-Time Sign Language Recognition

## 📌 Project Overview

This project is a **real-time American Sign Language (ASL) recognition system** that uses deep learning and computer vision to recognize hand signs through a webcam.

The system uses **MobileNetV2 transfer learning** for image classification and **MediaPipe Hands** for real-time hand detection. Recognized signs are converted into letters and combined into words or sentences. The system also includes **text-to-speech functionality** to speak the generated sentence.

## 🎯 Features

- Real-time ASL sign recognition using a webcam
- Recognition of **26 alphabet signs (A-Z)**
- Hand detection and tracking using MediaPipe
- MobileNetV2-based deep learning classifier
- Two-phase training with transfer learning and fine-tuning
- Temporal smoothing to reduce unstable predictions
- Sign-locking mechanism to prevent duplicate letters
- Sentence formation using recognized signs
- Backspace and clear functionality
- Text-to-speech output
- Real-time prediction confidence display

## 🧠 Model Architecture

The project uses **MobileNetV2 pretrained on ImageNet** as the backbone.

The initial training phase freezes the MobileNetV2 backbone and trains the classification layers. The second phase fine-tunes the later layers of the backbone with a lower learning rate. :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}

### Classification Head

```text
MobileNetV2
     ↓
Global Average Pooling
     ↓
Dense (128, ReLU)
     ↓
Dropout (0.3)
     ↓
Dense (64, ReLU)
     ↓
Dropout (0.2)
     ↓
Dense (26, Softmax)
     ↓
ASL Letter Prediction
