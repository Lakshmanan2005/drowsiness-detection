import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import threading
from playsound import playsound

import pygame

pygame.mixer.init()
sound = pygame.mixer.Sound("alarm.wav")

# Load model
model = load_model("model/drowsiness_model.h5")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                     'haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                    'haarcascade_eye.xml')

alarm_on = False

def play_alarm():
    playsound("alarm1.wav")

st.sidebar.title("About")
st.sidebar.info("This app detects driver drowsiness using deep learning and OpenCV.")

st.title("🚗 Driver Drowsiness Detection System")

run = st.checkbox('Start Camera')

frame_window = st.image([])

cap = cv2.VideoCapture(0)

counter = 0

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Camera not working")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex,ey,ew,eh) in eyes:
            eye = roi_gray[ey:ey+eh, ex:ex+ew]
            eye = cv2.resize(eye, (24,24))
            eye = eye / 255.0
            eye = eye.reshape(1,24,24,1)

            prediction = model.predict(eye)

            if np.argmax(prediction) == 1:
                counter += 1
            else:
                counter = 0

            if counter > 3:
                if not alarm_on:
                    alarm_on = True
                    sound.play()
            else:
                alarm_on = False

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame)

cap.release()