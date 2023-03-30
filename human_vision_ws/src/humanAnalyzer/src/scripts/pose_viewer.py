#!/usr/bin/env python3
from time import sleep
from typing import Tuple
import cv2
import mediapipe as mp
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from pose_detection.msg import pose_positions
from geometry_msgs.msg import Point
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

class PoseViewer:

    def __init__(self):
        self.received_image = None
        self.received_pose = None
        rospy.init_node('pose_viewer')
        rospy.loginfo('pose received')
        self.bridge = CvBridge()
        rospy.Subscriber('image', Image, self.process_image, queue_size=10)
        rospy.Subscriber('pose', pose_positions, self.process_pose, queue_size=10)
    
    def process_image(self, msg):
        try:
            self.received_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        except Exception as e:
            print(e)

    def process_pose(self, msg):
        self.received_pose = msg

    def show_image(img):
        cv2.imshow('image', img)
        cv2.waitKey(1)

    def run(self):
        while not rospy.is_shutdown():
            if self.received_image is not None and self.received_pose is not None:
                img_h, img_w, _ = self.received_image.shape
                showimg = self.received_image.copy()
                # get received pose attributes names
                #print(dir(self.received_pose))
                print(self.received_pose)
                print(self.received_pose.chest.x)
                showimg = cv2.circle(showimg, (int(self.received_pose.chest.x * img_w), int(self.received_pose.chest.y * img_h)), 5, (0, 255, 0), -1)
                #for hipRight
                showimg = cv2.circle(showimg, (int(self.received_pose.hipRight.x * img_w), int(self.received_pose.hipRight.y * img_h)), 5, (255, 255, 0), -1)
                #for hipLeft
                showimg = cv2.circle(showimg, (int(self.received_pose.hipLeft.x * img_w), int(self.received_pose.hipLeft.y * img_h)), 5, (0, 255, 255), -1)
                #for pose_point in self.received_pose:
                 #   cv2.circle(showimg, (int(pose_point.x), int(pose_point.y)), 5, (0, 255, 0), -1)
                cv2.imshow('image', showimg)
                cv2.waitKey(1)
                
            else:
                if self.received_image is None:
                    print("Waiting for image")
                if self.received_pose is None:
                    print("Waiting for pose")
            rospy.Rate(30).sleep()
                

if __name__ == '__main__':
    PoseViewer().run()