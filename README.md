# Driver Drowsiness Detection System

# Problem Statement
Driver drowsiness is a major cause of road accidents. This project aims to detect driver drowsiness in real-time using computer vision techniques and alert the driver to prevent accidents.

# Objective
To build a real-time system that monitors eye movements and detects signs of drowsiness using a webcam.


# Dataset
- Dataset Used: Eye State Dataset
- Source: Kaggle / OpenCV Haar Cascade
- Classes:
  - Open Eyes
  - Closed Eyes



# Tools & Technologies
- Python
- OpenCV
- TensorFlow / Keras
- NumPy
- CNN (Convolutional Neural Network)


# Workflow

1. Collect image dataset
2. Preprocess images
3. Train CNN model
4. Detect face using Haar Cascade
5. Detect eye state (Open/Closed)
6. Calculate drowsiness score
7. Trigger alarm if eyes remain closed


# Model Used
- Convolutional Neural Network (CNN)
- Input Shape: (24, 24, 1)
- Activation: ReLU
- Output: Softmax

# Results

- Model Accuracy: **XX%**
- Real-time detection working successfully
- Alarm triggered when drowsiness detected


# Output Screenshots

(Add images inside images folder)

Example:

![Detection Output](images/output1.png)


# Key Features

✅ Real-time webcam detection  
✅ CNN-based classification  
✅ Alarm alert system  
✅ Lightweight model  


# Future Improvements

- Add head pose detection
- Improve model accuracy
- Deploy as mobile application

# Project Structure
drowsiness-detection/
│
├── dataset/
├── model/
├── images/
├── drowsiness.py
├── requirements.txt
└── README.md


# Conclusion

This project successfully detects driver drowsiness using deep learning techniques and helps reduce accident risks by providing timely alerts.
