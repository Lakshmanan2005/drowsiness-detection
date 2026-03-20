import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical


data_dir = "dataset/train"

labels = ["open", "close"]
data = []
target = []

for label in labels:
    path = os.path.join(data_dir, label)
    class_num = labels.index(label)

    for img in os.listdir(path):
        try:
            img_path = os.path.join(path, img)
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            resized = cv2.resize(img_array, (24, 24))
            data.append(resized)
            target.append(class_num)
        except:
            pass

data = np.array(data).reshape(-1, 24, 24, 1) / 255.0
target = to_categorical(target, 2)

# Build model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(data, target, epochs=10, batch_size=32)

model.save("model/drowsiness_model.h5")

print("Model trained and saved!")