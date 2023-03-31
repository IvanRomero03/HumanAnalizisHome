#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import cv2
import numpy as np
import os
from time import sleep
#from deepface import DeepFace
from typing import TypedDict, List
from util_types.detection import DetectionResult, Facial_area
from util_types.recognition import FaceRecognitionRow
from humanAnalyzer.msg import pose_positions    
from deepface.deepface import DeepFace 
import time
import json

# disable gpu
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class faceRecognition:
    def __init__(self):
        self.prev_detections = {}
        self.faces_tracked = []
        self.margin = 10
        self.prev_frame_time = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontsize = 0.5

        self.faces_path = "src/humanAnalyzer/src/scripts/faces"
        self.img_counter = len(os.listdir(self.faces_path))
        self.json_path = "src/humanAnalyzer/src/scripts/identities.json"
        self.representations_path = "src/humanAnalyzer/src/scripts/faces/representations_vgg_face.pkl"

        #self.img_couter = len(os.listdir("faces"))
        self.received_image = None
        self.bridge = CvBridge()

        rospy.init_node('PoseDetector')

        self.imageSub = rospy.Subscriber(
            'image', Image, self.process_image, queue_size=10)

    def process_image(self, msg):
        try:
            self.received_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        except Exception as e:
            print("Image not received")
            print(e)

    def show_image(img):
        cv2.imshow('image', img)
        cv2.waitKey(1)
    
    def detect_faces(self, frame, detector_backend = 'ssd'):
        # detect faces
        imgFrame = np.asarray(frame)
        margin = self.margin
        try:
            faces_results: List[DetectionResult] = DeepFace.extract_faces(imgFrame, detector_backend = 'ssd')
        except:
            print("No faces detected")
            return None, None, None
        
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
        return faces, bboxes, xy_list 
    
    def draw_prev_detections(self, frame):
        prev_detections = self.prev_detections
        for prev_detection in prev_detections:
            #drawing previous detections
            prev_detection_bbox = prev_detections[prev_detection]
            p1 = (int(prev_detection_bbox[0]), int(prev_detection_bbox[1]))
            p2 = (int(prev_detection_bbox[0] + prev_detection_bbox[2]), int(prev_detection_bbox[1] + prev_detection_bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    
    def show_fps(self, frame, ):
        # font which we will be using to display FPS
        font = cv2.FONT_HERSHEY_SIMPLEX
        # time when we finish processing for this frame
        new_frame_time = time.time()
        # Calculating the fps
    
        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = 1/(new_frame_time-self.prev_frame_time)
        self.prev_frame_time = new_frame_time
        # converting the fps into integer
        fps = int(fps)
        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps) + ' fps'
        # putting the FPS count on the frame
        cv2.putText(frame, fps, (7, 20), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def check_prev_detections(self, frame, face):
        face_tracked = False
        prev_detections = self.prev_detections
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
                cv2.putText(frame, face_id, (face[0][0], face[0][1]), self.font, self.fontsize, (100, 255, 0), 1, cv2.LINE_AA)
                # updating tracker
                prev_detections[prev_detection] = face[2]
                print(f"appending {face_id} to faces tracked")
                self.faces_tracked.append(face_id)
                face_tracked = True
                break 
        return face_tracked
    
    def findFace(self, face, frame):
        find_result:List[FaceRecognitionRow] = DeepFace.find(face[1], db_path = self.faces_path, enforce_detection=False)[0]
        if(len(find_result) > 0):
            print("Face found")
            #print(find_result)
            print(face[0])
            face_id = find_result['identity'][0]
            if face_id not in self.prev_detections:
                #Creating tracker for face with identity face_id
                print("adding tracker")
                self.prev_detections[face_id] = face[2]
                self.faces_tracked.append(face_id)
                print(self.prev_detections)
                print(f"init tracker with id {face_id}")

            #getting attributes and writing to json
            #load json
            data = {}
            try:
                f = open(self.json_path)
                data = json.load(f)
            except:
                print("Created json file")
            
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
                    with open(self.json_path, 'w') as outfile:
                        json.dump(data, outfile)
                except:
                    print("error getting attributes")
                    self.faces_tracked.remove(face_id)
            else:
                print("face already in json")
            cv2.putText(frame, face_id, (face[0][0], face[0][1]), self.font, self.fontsize, (100, 255, 0), 1, cv2.LINE_AA)
            print("text added to frame")
        else:
            print("face not found")
            # save face to database
            cv2.imwrite(f"{self.faces_path}/face_{self.img_counter}.jpg", face[1])
            self.img_counter += 1
            # erase .pkl file
            os.remove(self.representations_path)
            #no find result (get data from face)

    def run(self):
        while not rospy.is_shutdown():
            if self.received_image is not None:
                frame = self.received_image
                # face detection, detector_backend can be changed
                faces, bboxes, xy_lists = self.detect_faces(frame, detector_backend = 'ssd')
                # Optional, to draw previous detection bounding box
                self.draw_prev_detections(frame)

                if faces == None:
                    #to avoid error
                    faces = []
                if(len(faces) > 0):
                    for face in zip(xy_lists, faces, bboxes):
                        # Check if face is being tracked, if not, run recognition
                        face_tracked = self.check_prev_detections(frame, face)
                        if not face_tracked:
                            self.findFace(face, frame)
                else:
                    # if no faces, empty faces tracked list
                    self.faces_tracked = []
                for prev_detection in self.prev_detections:
                    if prev_detection not in self.faces_tracked:
                        #deleting tracker
                        print(f"deleting tracker with id {prev_detection}")
                        del self.prev_detections[prev_detection]
                        break

                self.show_fps(frame)
                cv2.imshow('image', frame)
                cv2.waitKey(1)

            else:
                print("Waiting for image")
                sleep(1)
                continue

faceRecognition().run()


''' 
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

'''