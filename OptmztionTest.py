import cv2
import os
import numpy as np
from deepface.deepface.DeepFace import addRepresentation, find, find_w_face , analyze, extract_faces
from typing import TypedDict, List
from util_types.recognition import FaceRecognitionRow
from util_types.detection import DetectionResult, Facial_area
import matplotlib.pyplot as plt
import tensorflow as tf

cam = cv2.VideoCapture(0)

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# num of images in faces folder
img_counter = len(os.listdir("faces"))

# margin in pxs to add to face area
margin = 10

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    # image to numpy array
    imgFrame = np.asarray(frame)

    # detect faces
    try:
        faces_results: List[DetectionResult] = extract_faces(imgFrame, detector_backend = 'ssd')
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
    
    # for i, face in enumerate(faces):
    #     print(face)
    #     frame_prev = np.asarray(face)
    #     cv2.imshow(f"face_{i}", frame_prev)
    repeated_faces = []
    new_faces = []

    if(len(faces) > 0):
        print("faces detected")
        for face in faces:
            # Check if face is in database
            find_raw_result = find(face, db_path = "faces", enforce_detection=False)[0]
            find_result:List[FaceRecognitionRow] = find(face, db_path = "faces", enforce_detection=False)[0]
            print ("------------------")
            print (find_result)
            print ("------------------")
            if str(find_result["identity"]) == "unknown":
                print("new face found")
                new_faces.append(face)
            else:
                print("Found face with identity: ", find_result["identity"])
                repeated_faces.append(face)
        
        for i, face in enumerate(repeated_faces):
            frame_prev = np.asarray(face)
            cv2.imshow(f"repeated face_{i}", frame_prev)

        for i, face in enumerate(new_faces):
            frame_prev = np.asarray(face)
            cv2.imshow(f"new face_{i}", frame_prev)
        # # Add new faces to database
        # for face in new_faces:
        #     # Add new face to database
        #     addRepresentation(face, "faces", "opencv_frame_{}".format(img_counter))
        #     img_counter += 1
            

        # # Show repeated faces
        # for face in repeated_faces:
        #     print(face["identity"])
        #     plt.imshow(face)
        #     plt.show()
            
        # # Show new faces
        # for face in new_faces:
        #     plt.imshow(face)
        #     plt.show()
    cv2.imshow("test", frame)

cam.release()

cv2.destroyAllWindows()