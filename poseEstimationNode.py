#!/usr/bin/env python3
from time import sleep
from typing import Tuple
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.cam = cv2.VideoCapture(0)
        #self.img = cv2.imread("apuntando2.jpg")

    def run(self):
        with self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            
            while self.cam.isOpened():
                success, image = self.cam.read()
                if success:
                    #image = self.img
                    results = pose.process(image)

                    if results.pose_landmarks:
                        if not(results.pose_landmarks.landmark[12] and results.pose_landmarks.landmark[11] and results.pose_landmarks.landmark[19] and results.pose_landmarks.landmark[20]):
                            print("Not all points found")
                            continue
                        x = (
                            results.pose_landmarks.landmark[12].x + results.pose_landmarks.landmark[11].x) / 2
                        y = (
                            results.pose_landmarks.landmark[12].y + results.pose_landmarks.landmark[11].y) / 2
                        z = (
                            results.pose_landmarks.landmark[12].z + results.pose_landmarks.landmark[11].z) / 2
                        # Chest> x, y ,z
                        pointing = 0
                        left_shoulder = results.pose_landmarks.landmark[11]
                        right_shoulder = results.pose_landmarks.landmark[12]
                        left_index = results.pose_landmarks.landmark[19]
                        right_index = results.pose_landmarks.landmark[20]
                        h = image.shape[0]
                        w = image.shape[1]

                        # VERBOSE
                        image = cv2.circle(image, (int(left_shoulder.x*w), int(left_shoulder.y*h)), 5, (0, 0, 255), -1)
                        image = cv2.putText(image, "left_shoulder", (int(left_shoulder.x*w), int(left_shoulder.y*h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        image = cv2.circle(image, (int(right_shoulder.x*w), int(right_shoulder.y*h)), 5, (0, 0, 255), -1)
                        image = cv2.putText(image, "right_shoulder", (int(right_shoulder.x*w), int(right_shoulder.y*h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        image = cv2.circle(image, (int(left_index.x*w), int(left_index.y*h)), 5, (0, 0, 255), -1)
                        image = cv2.putText(image, "left_index", (int(left_index.x*w), int(left_index.y*h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        image = cv2.circle(image, (int(right_index.x*w), int(right_index.y*h)), 5, (0, 0, 255), -1)
                        image = cv2.putText(image, "right_index", (int(right_index.x*w), int(right_index.y*h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        m = 0.1
                        
                        right_out = right_index.x*w < right_shoulder.x*w - m*w 
                        left_out = left_index.x*w > left_shoulder.x*w + m*w

                        if right_out and left_out :
                            pointing = 2
                        elif right_out:
                            pointing = 0
                        elif left_out:
                            pointing = 1
                        else:
                            pointing = 2
                        if pointing == 0:
                            color = (0, 0, 255)
                        elif pointing == 1:
                            color = (0, 255, 0)
                        else:
                            color = (255, 0, 0)
                        
                        # VERBOSE
                        image = cv2.rectangle(image, (int(left_shoulder.x*w + m*w), 0), (int(right_shoulder.x*w - m*w), h), color, 3)

                        cv2.imshow('MediaPipe Pose', image)
                        if cv2.waitKey(1) & 0xFF == 27:
                            break

asd = PoseDetector()
asd.run()