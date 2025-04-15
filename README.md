# 🚗 Driver Monitoring System

A real-time Computer Vision system for detecting driver drowsiness and attention using deep learning and Mediapipe. Achieves 98% validation accuracy for eye state detection.

## 🔍 Project Overview

This project uses computer vision and machine learning to monitor a driver's eye status and head pose in real-time. The goal is to detect drowsiness or distracted behavior and trigger safety alerts.

##  Key Features

- Eye state detection (open/closed) with CNN
- Head pose estimation using Mediapipe
- Real-time video inference on Raspberry Pi 4
- Optimized deployment using C++ and Mediapipe Lite
- HMI interface for data display and feedback

##  Tech Stack

- Python, C++
- Mediapipe, OpenCV
- TensorFlow
- Raspberry Pi 4
- HMI interface (custom)

##  Results

- 98% validation accuracy
- 97% training accuracy
- Real-time performance on low-resource device

##  Installation

```bash
git clone https://github.com/AliN3bil2/driver-monitoring-system.git
cd driver-monitoring-system
pip install -r requirements.txt
python main.py
