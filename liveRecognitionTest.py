import cv2
import numpy as np
import os
from deepface.deepface import DeepFace
from typing import TypedDict, List
from util_types.detection import DetectionResult, Facial_area
from util_types.recognition import FaceRecognitionRow
import time

# cam 
cam = cv2.VideoCapture(0)

# used to record the time when we processed last frame
prev_frame_time = 0
  
# used to record the time at which we processed current frame
new_frame_time = 0

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
    xy_list = []
    for face in faces_results:
        face_area = face["facial_area"]
        face_img = imgFrame[max(0, face_area["y"]-margin):min(imgFrame.shape[0], face_area["y"]+face_area["h"]+margin), max(0, face_area["x"]-margin):min(imgFrame.shape[1], face_area["x"]+face_area["w"]+margin)]
        faces.append(face_img)
        xy_list.append((face_area["x"], face_area["y"]))
        cv2.rectangle(frame, (max(0, face_area["x"]-margin), max(0, face_area["y"]-margin)), (min(imgFrame.shape[1], face_area["x"]+face_area["w"]+margin), min(imgFrame.shape[0], face_area["y"]+face_area["h"]+margin)), (0, 255, 0), 2)

    repeated_faces = []
    new_faces = []

    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()

    # Calculating the fps
  
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
  
    # converting the fps into integer
    fps = int(fps)
  
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps) + ' fps'
  
    # putting the FPS count on the frame
    cv2.putText(frame, fps, (7, 70), font, 1, (100, 255, 0), 3, cv2.LINE_AA)

    if(len(faces) > 0):
        print("faces detected")
        for face in zip(xy_list, faces):
            # Check if face is in database
            find_result:List[FaceRecognitionRow] = DeepFace.find(face[1], db_path = "faces", enforce_detection=False)[0]
            if(len(find_result) > 0):
                print("face found")
                #print(find_result)
                print(face[0])
                cv2.putText(frame, find_result['identity'][0], (face[0][0], face[0][1]), font, 1, (100, 255, 0), 1, cv2.LINE_AA)
            else:
                print("face not found")
                # save face to database
                cv2.imwrite("faces/face_{}.jpg".format(img_counter), face[1])
                img_counter += 1
                # erase .pkl file
                os.remove("faces/representations_vgg_face.pkl")
                #no find result (get data from face)
    cv2.imshow("test", frame)


cam.release()