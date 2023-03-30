#!/usr/bin/env python3
from time import sleep
from typing import Tuple
import cv2
import mediapipe as mp
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from humanAnalyzer.msg import pose_positions
from geometry_msgs.msg import Point

# indexToName = ["nose",
#                "leftEyeInner",
#                "leftEye",
#                "leftEyeOuter",
#                "rightEyeInner",
#                "rightEye",
#                "rightEyeOuter",
#                "leftEar",
#                "rightEar",
#                "mouthLeft",
#                "mouthRight",
#                "leftShoulder",
#                "rightShoulder",
#                "leftElbow",
#                "rightElbow",
#                "leftWrist",
#                "rightWrist",
#                "leftPinky",
#                "rightPinky",
#                "leftIndex",
#                "rightIndex",
#                "leftThumb",
#                "rightThumb",
#                "leftHip",
#                "rightHip",
#                "leftKnee",
#                "rightKnee",
#                "leftAnkle",
#                "rightAnkle",
#                "leftHeel",
#                "rightHeel",
#                "leftFootIndex",
#                "rightFootIndex"]


PublisherPoints = [
    {"name": "shoulderLeft", "index": 11},
    {"name": "shoulderRight", "index": 12},
    {"name": "elbowLeft", "index": 13},
    {"name": "elbowRight", "index": 14},
    {"name": "wristLeft", "index": 15},
    {"name": "wristRight", "index": 16},
    {"name": "pinkyLeft", "index": 17},
    {"name": "pinkyRight", "index": 18},
    {"name": "indexLeft", "index": 19},
    {"name": "indexRight", "index": 20},
    {"name": "thumbLeft", "index": 21},
    {"name": "thumbRight", "index": 22},
    {"name": "hipLeft", "index": 23},
    {"name": "hipRight", "index": 24},
    # {"name": "chest", "index": 33},
]


class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.imageReceved = None

        self.bridge = CvBridge()

        rospy.init_node('PoseDetector')

        self.imageSub = rospy.Subscriber(
            'image', Image, self.image_callback, queue_size=10)
        print(type(self.imageSub))

        self.posePub = rospy.Publisher(
            "pose", pose_positions, queue_size=10)

    def image_callback(self, data):
        self.imageReceved = data

    def run(self):
        with self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            while not rospy.is_shutdown():
                if self.imageReceved is not None:
                    image = self.bridge.imgmsg_to_cv2(
                        self.imageReceved, "rgb8")
                    image.flags.writeable = False
                    results = pose.process(image)

                    if results.pose_landmarks:
                        x = (
                            results.pose_landmarks.landmark[12].x + results.pose_landmarks.landmark[11].x) / 2
                        y = (
                            results.pose_landmarks.landmark[12].y + results.pose_landmarks.landmark[11].y) / 2
                        z = (
                            results.pose_landmarks.landmark[12].z + results.pose_landmarks.landmark[11].z) / 2
                        posePublish = pose_positions()
                        
                        for(i, landmark) in enumerate(results.pose_landmarks.landmark[11:25]):
                            point = Point()
                            initName = PublisherPoints[i]["name"]
                            point.x = landmark.x
                            point.y = landmark.y
                            point.z = landmark.z
                            posePublish.__setattr__(initName, point)
                        point = Point()
                        point.x = x
                        point.y = y
                        point.z = z

                        posePublish.chest = point
                        self.posePub.publish(posePublish)
                        sleep(0.1)
                else:
                    print("Image not received")
                rospy.Rate(30).sleep()


PoseDetector().run()