import cv2
import numpy as np
import os
from deepface.deepface import DeepFace
from typing import TypedDict, List
from util_types.detection import DetectionResult, Facial_area
from util_types.recognition import FaceRecognitionRow
import time
import json

# opencv version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# disable gpu
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# cam 
cam = cv2.VideoCapture(0)

# set tracking type
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[1]

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

# num of images in faces folder
img_counter = len(os.listdir("faces"))

# margin in pxs to add to face area
margin = 10

trackers = {}
prev_detections = {}
faces_tracked = []

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
        faces_results: List[DetectionResult] = DeepFace.extract_faces(imgFrame, detector_backend = 'ssd')
    except:
        print("no faces detected")
        continue
    
    # dict_keys(['face', 'facial_area', 'confidence'])
    faces = []
    bboxes = []
    xy_list = []
    for face in faces_results:
        #detecting faces
        face_area = face["facial_area"]
        face_img = imgFrame[max(0, face_area["y"]-margin):min(imgFrame.shape[0], face_area["y"]+face_area["h"]+margin), max(0, face_area["x"]-margin):min(imgFrame.shape[1], face_area["x"]+face_area["w"]+margin)]
        faces.append(face_img)
        xy_list.append((face_area["x"], face_area["y"]))
        bbox = (max(0, face_area["x"]-margin), max(0, face_area["y"]-margin), face_area["w"]+margin, face_area["h"]+margin)
        bboxes.append(bbox)
        cv2.rectangle(frame, (max(0, face_area["x"]-margin), max(0, face_area["y"]-margin)), (min(imgFrame.shape[1], face_area["x"]+face_area["w"]+margin), min(imgFrame.shape[0], face_area["y"]+face_area["h"]+margin)), (0, 255, 0), 2)

    for prev_detection in prev_detections:
        #drawing previous detections
        prev_detection_bbox = prev_detections[prev_detection]
        p1 = (int(prev_detection_bbox[0]), int(prev_detection_bbox[1]))
        p2 = (int(prev_detection_bbox[0] + prev_detection_bbox[2]), int(prev_detection_bbox[1] + prev_detection_bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
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
        faces_tracked = []
        for face in zip(xy_list, faces, bboxes):
            face_tracked = False
            #checking if face detected is being tracked
            for prev_detection in prev_detections:
                face_id = prev_detection
                prev_detection_bbox = prev_detections[prev_detection]
                margin_w = prev_detection_bbox[2] * 0.3
                margin_h = prev_detection_bbox[3] * 0.3
                tracker_x = prev_detection_bbox[0]
                tracker_y = prev_detection_bbox[1]
                tracker_w = prev_detection_bbox[2]
                tracker_h = prev_detection_bbox[3]
                face_x = face[2][0]
                face_y = face[2][1]
                face_w = face[2][2]
                face_h = face[2][3]
                if ((tracker_x - margin_w < face_x < tracker_x + margin_w) and 
                    (tracker_y - margin_h < face_y < tracker_y + margin_h)):
                    face_tracked = True
                    cv2.putText(frame, face_id, (face[0][0], face[0][1]), font, 1, (100, 255, 0), 1, cv2.LINE_AA)
                    # updating tracker
                    prev_detections[prev_detection] = face[2]
                    print(f"appending {face_id} to faces tracked")
                    faces_tracked.append(face_id)
                
            #If face is not being tracked, try to find it in database
            if not face_tracked:
                find_result:List[FaceRecognitionRow] = DeepFace.find(face[1], db_path = "faces", enforce_detection=False)[0]
                if(len(find_result) > 0):
                    print("face found")
                    #print(find_result)
                    print(face[0])
                    face_id = find_result['identity'][0]
                    if face_id not in prev_detections:
                        #Creating tracker for face with identity face_id
                        print("adding tracker")
                        prev_detections[face_id] = face[2]
                        faces_tracked.append(face_id)
                        print(prev_detections)
                        print(f"init tracker with id {face_id}")

                    #getting attributes and writing to json
                    #load json
                    f = open('identities.json')
                    data = json.load(f)
                    if face_id not in data:
                        try:
                            features_list = DeepFace.analyze(face[1], enforce_detection=True)
                            print(f"features list size is {len(features_list)}")
                            features = features_list[0]
                            age = features.get('age')
                            gender = features.get('dominant_gender')
                            race = features.get('dominant_race')

                            data[face_id] = {
                                "age": age,
                                "gender": gender,
                                "race": race
                                }
                            with open('identities.json', 'w') as outfile:
                                json.dump(data, outfile)
                        except:
                            print("error getting attributes")
                            faces_tracked.remove(face_id)

                    cv2.putText(frame, face_id, (face[0][0], face[0][1]), font, 1, (100, 255, 0), 1, cv2.LINE_AA)
                    print("text added to frame")
                else:
                    print("face not found")
                    # save face to database
                    cv2.imwrite("faces/face_{}.jpg".format(img_counter), face[1])
                    img_counter += 1
                    # erase .pkl file
                    os.remove("faces/representations_vgg_face.pkl")
                    #no find result (get data from face)
        for prev_detection in prev_detections:
            if prev_detection not in faces_tracked:
                #deleting tracker
                print(f"deleting tracker with id {prev_detection}")
                del prev_detections[prev_detection]
                break
    cv2.imshow("test", frame)


cam.release()