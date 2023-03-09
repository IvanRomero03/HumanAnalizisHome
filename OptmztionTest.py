import cv2
import os
import numpy as np
from deepface.deepface.DeepFace import addRepresentation, find_w_face , analyze, extract_faces
from typing import TypedDict, List
from util_types.recognition import FaceRecognitionRow
from util_types.detection import DetectionResult, Facial_area
import matplotlib.pyplot as plt

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

    # image to numpy array
    imgFrame = np.asarray(frame)

    # detect faces
    try:
        faces_results: List[DetectionResult] = extract_faces(imgFrame, detector_backend = 'opencv')
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
            find_result:List[FaceRecognitionRow] = find_w_face(face, db_path = "faces", enforce_detection=False)[0]
            if find_result["identity"] == "unknown":
                new_faces.append(face)
            else:
                repeated_faces.append(find_result)
            
        # Add new faces to database
        for face in new_faces:
            # Add new face to database
            addRepresentation(face, "faces", "opencv_frame_{}".format(img_counter))
            img_counter += 1
            

        # Show repeated faces
        for face in repeated_faces:
            print(face["identity"])
            plt.imshow(face["face"])
            plt.show()
            
        # Show new faces
        for face in new_faces:
            plt.imshow(face)
            plt.show()

cam.release()

cv2.destroyAllWindows()

