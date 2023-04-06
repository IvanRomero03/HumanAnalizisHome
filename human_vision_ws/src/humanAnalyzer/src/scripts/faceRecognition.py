#!/usr/bin/env python3
import rospy
import rospkg
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
from humanAnalyzer.msg import face, face_array
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
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('humanAnalyzer')
        self.faces_path = f"{pkg_path}/src/scripts/faces"
        self.img_counter = len(os.listdir(self.faces_path))
        self.json_path = f"{pkg_path}/src/scripts/identities.json"
        self.representations_path = f"{pkg_path}/src/scripts/faces/representations_vgg_face.pkl"

        #self.img_couter = len(os.listdir("faces"))
        self.received_image = None
        self.bridge = CvBridge()

        rospy.init_node('PoseDetector')

        self.imageSub = rospy.Subscriber(
            'image', Image, self.process_image, queue_size=10)
        self.facePub = rospy.Publisher(
            'faces', face_array, queue_size=10
        )

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
            print(f"checking prev detection: {prev_detection}")
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
        if self.img_counter > 0:
            find_result:List[FaceRecognitionRow] = DeepFace.find(face[1], db_path = self.faces_path, enforce_detection=False)[0]
            if(len(find_result) > 0):
                print("Face found")
                #print(find_result)
                print(face[0])
                face_id_path = find_result['identity'][0]
                path_to_face, face_id = os.path.split(face_id_path)
                print(f"face id is {face_id}")

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
                        del self.prev_detections[face_id]
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
        else:
            print("no faces in path")
            # save face to database
            cv2.imwrite(f"{self.faces_path}/face_{self.img_counter}.jpg", face[1])
            self.img_counter += 1

    def publishFaces(self):
        faces_publish = []
        if self.prev_detections:
            for i, prev_detection in enumerate(self.prev_detections):
                face_id = prev_detection
                bbox = self.prev_detections[prev_detection]
                face_publish = face()
                face_publish.identity = face_id
                face_publish.x = bbox[0]
                face_publish.y = bbox[1]
                face_publish.w = bbox[2]
                face_publish.h = bbox[3]
                faces_publish.append(face_publish)
            print(f"published {faces_publish}")
            self.facePub.publish(faces_publish)

    def run(self):
        while not rospy.is_shutdown():
            if self.received_image is not None:
                print("----------------------------------------------------")
                frame = self.received_image
                # face detection, detector_backend can be changed
                faces, bboxes, xy_lists = self.detect_faces(frame, detector_backend = 'ssd')
                # Optional, to draw previous detection bounding box
                self.draw_prev_detections(frame)
                self.faces_tracked = []

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
                self.publishFaces()
                cv2.imshow('image', frame)
                cv2.waitKey(1)

            else:
                print("Waiting for image")
                sleep(1)
                continue

faceRecognition().run()