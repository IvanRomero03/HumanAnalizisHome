#!/usr/bin/env python3

# Asynchronous service that runs deepface age, gender and race models to store on JSON file
import rospy
import rospkg
from humanAnalyzer.srv import imagetoAnalyze, imagetoAnalyzeResponse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Int16, Bool
from deepface.deepface import DeepFace 
import json
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Toggle to run Deepface on GPU or CPU
useGPU = False

class faceAnalysis:

    def __init__(self):
        rospy.init_node('faceAnalysis')
        self.doAction = False
        # Setting as busy to initialize
        self.statePub = rospy.Publisher("faceAnalysisState", Bool, queue_size=1)
        self.statePub.publish(True)

        # Setting up image receiver
        self.received_image = None
        self.bridge = CvBridge()
        self.imageSub = rospy.Subscriber(
            'image', Image, self.process_image, queue_size=10)
        
        # Setting up server
        self.analysisSrv = rospy.Service('faceAnalysisSrv', imagetoAnalyze, self.handle_request)
        
        # JSON location
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('humanAnalyzer')
        self.json_path = f"{pkg_path}/src/scripts/identities.json"

        # Loading deepface models
        print("Loading deepface models...")
        DeepFace.build_model("Age")
        DeepFace.build_model("Race")
        DeepFace.build_model("Gender")
        print("Deepface models loaded")

    # Receive image from camera
    def process_image(self, msg):
        try:
            self.received_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        except Exception as e:
            print("Image not received")
            print(e)
    
    #-----------SERVER---------------------------------
    def handle_request(self, req):
        print(f"Received request from client : {req}")
        self.req = req
        self.doAction = True
        return True

    
    def analysisProcess(self):
        if self.doAction:
            self.statePub.publish(True)
            # Running face analysis
            print("Running face analysis...")
            face_id = self.req.imagetoAnalyze.identity
            face_x = self.req.imagetoAnalyze.x
            face_y = self.req.imagetoAnalyze.y
            face_w = self.req.imagetoAnalyze.w
            face_h = self.req.imagetoAnalyze.h
            faceimg = self.received_image[face_y:face_y+face_h, face_x:face_x+face_w]
            #show faceimg in plt
            #plt.imshow(cv2.cvtColor(faceimg, cv2.COLOR_BGR2RGB))
            #plt.show()
            data = {}
            try:
                f = open(self.json_path)
                data = json.load(f)
                if "age" not in data[face_id]:
                    try:
                        features_list = DeepFace.analyze(self.received_image, enforce_detection=True)
                        features = features_list[0]
                        age = features.get('age')
                        gender = features.get('dominant_gender')
                        race = features.get('dominant_race')
                        data[face_id]["age"] = age
                        data[face_id]["gender"] = gender
                        data[face_id]["race"] = race
                        with open(self.json_path, 'w') as outfile:
                            json.dump(data, outfile)
                    except:
                        print("Error getting attributes")
                else:
                    print(f"Attributes for {face_id} already in json")
            except:
                print("Could not open JSON ")
            '''
            data = {}
            try:
                f = open(self.json_path)
                data = json.load(f)
                if face_id not in data:
                    try:
                        features_list = DeepFace.analyze(self.received_image, enforce_detection=True)
                        features = features_list[0]
                        age = features.get('age')
                        gender = features.get('dominant_gender')
                        race = features.get('dominant_race')
                        data[face_id]["age"] = age

                        
                        with open(self.json_path, 'w') as outfile:
                            json.dump(data, outfile)
                    except:
                        print("Error getting attributes")
                else:
                    print("face already in json")
            except:
                print("Could not open JSON ")
            '''
            print("Face analysis finished")
            self.doAction = False
            return True
            

    def run(self):
        
        print("Ready to receive requests")
        while not rospy.is_shutdown():
            self.statePub.publish(False)
            if self.received_image is not None:
                self.analysisProcess()
            rospy.Rate(5.0).sleep()

if useGPU:
    faceAnalysis().run()
else:
    with tf.device('/cpu:0'):
        faceAnalysis().run()
