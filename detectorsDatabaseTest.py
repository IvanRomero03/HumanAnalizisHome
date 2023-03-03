import cv2
import numpy as np
import os
import time
from deepface import DeepFace
from typing import TypedDict, List

# Checking correct version
print(DeepFace.__file__)

detector_backends = ["opencv", "ssd", "mtcnn", "retinaface"]
detector_times = {}

database_path = "detectorTest_db_small"

margin = 10

# num of images in faces folder
img_counter = len(os.listdir(database_path))

for backend in detector_backends:
    print(f"Testing {backend} backend")
    tick = time.time()
    #creating database folder
    if (not os.path.exists(f"{backend}_extracted")):
            print(f"Had to create {backend}_extracted folder")
            os.mkdir(f"{backend}_extracted")
    else:
         for file in os.listdir(f"{backend}_extracted"):
            os.remove(f"{backend}_extracted/{file}")

    for i in range(img_counter):
        db_img = cv2.imread(f"{database_path}/face_{i}.jpg")
        #extracting faces
        try:
            face_objs = DeepFace.extract_faces(
                db_img, detector_backend=backend
            )
            #saving faces
            for face_count, face_obj in enumerate(face_objs):
                face_area = face_obj["facial_area"]
                face_img = db_img[max(0, face_area["y"]-margin):min(db_img.shape[0], face_area["y"]+face_area["h"]+margin), max(0, face_area["x"]-margin):min(db_img.shape[1], face_area["x"]+face_area["w"]+margin)]
                cv2.imwrite(f"{backend}_extracted/face_{i}_{face_count}_{backend}.jpg", face_img)
        except:
            print(f"Error in {backend} backend with face_{i}.jpg")
            continue

    #counting time
    detector_times[backend] = time.time() - tick
    print(f"Done testing {backend} backend in {detector_times[backend]} seconds")

print(detector_times)

