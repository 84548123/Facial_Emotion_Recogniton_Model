# 🎭 Real-Time Facial Emotion Recognition

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📖 Overview

This project is a deep learning application capable of detecting human faces from a live video feed and classifying their emotions in real-time.

Built using **Convolutional Neural Networks (CNN)** and **OpenCV**, the system recognizes 7 distinct emotional states: **Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral**. It is designed to be lightweight and fast, making it suitable for real-time interaction.

---

## ✨ Key Features

* **⚡ Real-Time Processing:** Detects and classifies emotions instantly from a webcam feed.
* **🧠 Deep Learning Model:** Custom CNN architecture trained on the **FER-2013** dataset.
* **📷 Face Detection:** Utilizes OpenCV Haar Cascades for robust face tracking.
* **📊 Accuracy:** Achieves competitive validation accuracy on the test set.
* **📉 Visualizations:** Includes scripts to plot training loss and accuracy curves.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Deep Learning:** TensorFlow / Keras
* **Computer Vision:** OpenCV (`cv2`)
* **Data Handling:** NumPy, Pandas
* **Visualization:** Matplotlib

---

## 🧠 Model Architecture

The model is a Convolutional Neural Network (CNN) designed for image classification. It consists of:
1.  **Convolutional Layers:** To extract spatial features (edges, textures).
2.  **Max Pooling:** To reduce dimensionality and computation.
3.  **Dropout Layers:** To prevent overfitting.
4.  **Dense Layers:** For final classification (Softmax output).



---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/emotion-recognition.git](https://github.com/yourusername/emotion-recognition.git)
cd emotion-recognition

emotion-recognition/
├── data/               # Dataset (FER-2013)
├── model/              # Saved model weights (.h5 file)
├── src/                # Source code
│   ├── train.py        # Script to train the CNN
│   ├── predict.py      # Real-time inference script
│   └── model.py        # CNN architecture definition
├── haarcascade_frontalface_default.xml # OpenCV face detector
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```bash
git clone [https://github.com/yourusername/emotion-recognition.git](https://github.com/yourusername/emotion-recognition.git)
cd emotion-recognition
