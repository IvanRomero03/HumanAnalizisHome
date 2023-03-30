#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def process_image(msg):
    try:
        bridge = CvBridge()
        orig = bridge.imgmsg_to_cv2(msg, "rgb8")
        print(f"read image")
        img = orig
        show_image(img)
    except Exception as e:
        print(e)

def show_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(1)

def img_listener():
    rospy.init_node('image_viewer')
    rospy.loginfo('image received')
    rospy.Subscriber('image', Image, process_image, queue_size=10)
    rospy.spin()

if __name__ == '__main__':
    try:
        img_listener()
    except rospy.ROSInterruptException:
        pass

