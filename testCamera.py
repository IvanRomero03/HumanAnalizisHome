import cv2
import numpy as np
import os
from deepface.deepface import DeepFace
from typing import TypedDict, List
from util_types.detection import DetectionResult, Facial_area
from util_types.recognition import FaceRecognitionRow
    
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

# margin in pxs to add to face area
margin = 10

while True: # try to get the first frame
    rval, frame = vc.read()
    # Capture frame-by-frame
    ret, frame = vc.read()
    
    #Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # image to numpy array
    #imgFrame = np.asarray(frame)
    imgFrame = np.asarray(gray)

    # detect faces
    try:
        faces_results: List[DetectionResult] = DeepFace.extract_faces(imgFrame, detector_backend = 'opencv')
    except:
        print("no faces detected")
        continue
    
     # dict_keys(['face', 'facial_area', 'confidence'])
    faces = []
    for face in faces_results:
        face_area = face["facial_area"]
        face_img = imgFrame[max(0, face_area["y"]-margin):min(imgFrame.shape[0], face_area["y"]+face_area["h"]+margin), max(0, face_area["x"]-margin):min(imgFrame.shape[1], face_area["x"]+face_area["w"]+margin)]
        faces.append(face_img)
        cv2.rectangle(frame, (max(0, face_area["x"]-margin), max(0, face_area["y"]-margin)), (min(imgFrame.shape[1], face_area["x"]+face_area["w"]+margin), min(imgFrame.shape[0], face_area["y"]+face_area["h"]+margin)), (0, 255, 0), 2)

    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("preview")