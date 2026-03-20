import cv2
import numpy as np
from tensorflow.keras.models import load_model
from playsound import playsound
import threading

model = load_model("model/drowsiness_model.h5")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                     'haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                    'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
counter = 0

import pygame

pygame.mixer.init()
sound = pygame.mixer.Sound("alarm1.wav")

def alarm():
    playsound("alarm1.wav")

while True:
    ret, frame = cap.read()
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

            if np.argmax(prediction) == 1:  # closed
                counter += 1
            else:
                counter = 0

            if counter > 3:
                cv2.putText(frame, "DROWSY!", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                threading.Thread(target=alarm).start()

                if not alarm_on:
                    alarm_on = True
                    threading.Thread(target=alarm).start()
            else:
                alarm_on = False

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()