# feb 17 2023

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import cv2
from deepface.deepface import DeepFace
import tensorflow as tf


os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
]

detector_backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface"]

img_path = "dataset/img1.jpg"

model = "Age"

print(f"Building model {model}")

cam = cv2.VideoCapture(0)


captured, image = cam.read()
print(type(image))

while captured:
    captured, image = cam.read()

    image = cv2.resize(image, (640, 480))

    cv2.imshow("Webcam", image)
    try:
        features_arr = DeepFace.analyze(img_path=image, actions=("age"), enforce_detection=True)
        print(features_arr)
        for features in features_arr:
            print("Age: ", features.get("age"))
            region = features.get("region")
            print("Finished scan")
            x_coord = region["x"]
            y_coord = region["y"]
            width = region["w"]
            height = region["h"]
            color = (0, 255, 0)
            image = cv2.rectangle(
                image, (x_coord, y_coord), (x_coord + width, y_coord + height), color, 2
            )
            cv2.putText(
                image,
                f'Age: {features.get("age")},',
                (x_coord, y_coord - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (36, 255, 12),
                2,
            )
    except:
        print("No face detected")

    key = cv2.waitKey(1)

    if key == 27:
        break
