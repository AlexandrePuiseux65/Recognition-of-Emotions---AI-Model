# 🎥 Live Emotion Recognition
![Status](https://img.shields.io/badge/Status-Work%20In%20Progress-yellow)

Real-time facial emotion detection using a custom CNN model and OpenCV.  
Detects **5 emotions** directly from a webcam stream.

## Description

This project is a personal upgrade of an undergraduate group project on emotion recognition. The original had limitations I wanted to address, so I rebuilt it from scratch with two goals:

- Train a CNN model from scratch targeting 55–65% accuracy on a 5-class emotion dataset (Last model had 37% since we didn't have access to a GPU cluster).
- Deploy it live using OpenCV's facial detection pipeline.

The model was trained on **84,878 images** from a Kaggle dataset, using a VGG-inspired architecture with BatchNormalization and Dropout for regularization.

## Model Details

- **Architecture**: Custom CNN inspired by VGG16 (3 conv blocks + dense layers)
- **Input**: 48×48 grayscale images
- **Classes**: `angry`, `happy`, `neutral`, `sad`, `surprise`
- **Test accuracy**: ~60.7% | F1 Score: ~61.2%
- **Framework**: TensorFlow / Keras


# Getting Started
## Dependencies

- Python 3.11
- A webcam

See `requirement.txt`. Key libraries: `tensorflow`, `opencv-python`, `numpy`, `keras`.

## Installation

```bash
# Clone the repo
git clone https://github.com/AlexandrePuiseux65/Recognition-of-Emotions---AI-Model.git
cd Recognition-of-Emotions---AI-Model

# (Recommended) Create a virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirement.txt
```

## Running the program

```bash
python weebcam/main.py
```

Press `ESC` to stop the stream.

## Authors
* [@Alexandre Puiseux](https://github.com/AlexandrePuiseux65)

# Next Steps

1. Improve detection of minority classes (`angry`, `sad`, `surprise`) via oversampling or stronger data augmentation
2. Investigate domain shift between training data and live webcam input
3. Export model to `.keras` format (replacing legacy `.h5`)
4. Add a confidence threshold to avoid displaying low-confidence predictions

# Acknowledgments

## Dataset
> Tiryaki, S. *Facial Expressions Classification*. Kaggle.  
> https://www.kaggle.com/code/sahintiryaki/facial-expressions-classification

## References
> Sam Westby — OpenCV-Python-Tutorial (facial detection example)  
> https://github.com/samwestby/OpenCV-Python-Tutorial/blob/main/6_facial_detection.py

> Briar Campbell et al. — *Check for Nvidia GPU in Python*, Stack Overflow, 2026-05-27  
> https://stackoverflow.com/a/67504607 — License: CC BY-SA 4.0
