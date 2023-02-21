import cv2
import numpy as np
import os
from deepface import DeepFace
from typing import TypedDict, List
from util_types.detection import DetectionResult, Facial_area
from util_types.recognition import FaceRecognitionRow

# cam 
cam = cv2.VideoCapture(0)

# num of images in faces folder
img_counter = len(os.listdir("faces"))

# margin in pxs to add to face area
margin = 10

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    # elif k%256 == 32:
    #     # SPACE pressed
    #     img_name = "opencv_frame_{}.png".format(img_counter)
    #     cv2.imwrite(img_name, frame)
    #     print("{} written!".format(img_name))
    #     img_counter += 1
    
    # image to numpy array
    imgFrame = np.asarray(frame)

    # detect faces
    try:
        faces_results: List[DetectionResult] = DeepFace.extract_faces(imgFrame, detector_backend = 'retinaface')
    except:
        print("no faces detected")
        continue
    
    # dict_keys(['face', 'facial_area', 'confidence'])
    faces = []
    for face in faces_results:
        face_area = face["facial_area"]
        face_img = imgFrame[max(0, face_area["y"]-margin):min(imgFrame.shape[0], face_area["y"]+face_area["h"]+margin), max(0, face_area["x"]-margin):min(imgFrame.shape[1], face_area["x"]+face_area["w"]+margin)]
        faces.append(face_img)

    repeated_faces = []
    new_faces = []

    if(len(faces) > 0):
        print("faces detected")
        for face in faces:
            # Check if face is in database
            find_result:List[FaceRecognitionRow] = DeepFace.find(face, db_path = "faces", enforce_detection=False)[0]
            if(len(find_result) > 0):
                print("face found")
                print(find_result)
            else:
                print("face not found")
                # save face to database
                cv2.imwrite("faces/face_{}.jpg".format(img_counter), face)
                img_counter += 1
                # erase .pkl file
                os.remove("faces/representations_vgg_face.pkl")
                #no find result (get data from face)
    
cam.release()