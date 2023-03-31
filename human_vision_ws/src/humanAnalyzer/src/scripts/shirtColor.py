#!/usr/bin/env python3
from time import sleep
from typing import Tuple
import cv2
import mediapipe as mp
import numpy as np
import rospy
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from humanAnalyzer.msg import pose_positions
from geometry_msgs.msg import Point
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

class shirtColor:

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
    
    def classifyColor(self, rgb):
        r = rgb[0]
        g = rgb[1]
        b = rgb[2]
         #Training data with color names and their RGB values
        colors = {'red': [255,0,0],
                'green': [0,255,0],
                'blue': [0,0,255],
                'yellow': [255,255,0],
                'cyan': [0,255,255],
                'magenta': [255,0,255],
                'white': [255,255,255],
                'black': [0,0,0],
                'gray': [128,128,128],
                'purple': [128,0,128],
                'orange': [255,165,0],
                'pink': [255,192,203],
                'brown': [165,42,42]}

        min_distance = float('inf')
        closest_color = None
        for color, rgb in colors.items():
            # get the smalles distance from obtained color to the set colors
            distance = math.sqrt((rgb[0]-r)**2 + (rgb[1]-g)**2 + (rgb[2]-b)**2)
            if distance < min_distance:
                min_distance = distance
                closest_color = color
        return closest_color
    
    def get_biggest_contour(self, img):
        R,G,B = cv2.split(img)

        # Do some denosiong on the red chnnale (The red channel gave better result than the gray because it is has more contrast
        Rfilter = cv2.bilateralFilter(R,25,25,10)

        # Threshold image
        ret, Ithres = cv2.threshold(Rfilter,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # Find the largest contour and extract it
        contours, contours2 = cv2.findContours(Ithres,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE )

        maxContour = 0
        for contour in contours:
            contourSize = cv2.contourArea(contour)
            if contourSize > maxContour:
                maxContour = contourSize
                maxContourData = contour

        # Create a mask from the largest contour
        mask = np.zeros_like(Ithres)
        cv2.fillPoly(mask,[maxContourData],1)

        # Use mask to crop data from original image
        finalImage = np.zeros_like(img)
        finalImage[:,:,0] = np.multiply(R,mask)
        finalImage[:,:,1] = np.multiply(G,mask)
        finalImage[:,:,2] = np.multiply(B,mask)

        return finalImage

    def run(self):
        while not rospy.is_shutdown():
            if self.received_image is not None and self.received_pose is not None:
                img_h, img_w, _ = self.received_image.shape
                showimg = self.received_image.copy()
                # get received pose attributes names
                #print(dir(self.received_pose))
                print(self.received_pose)
                showimg = cv2.circle(showimg, (int(self.received_pose.chest.x * img_w), int(self.received_pose.chest.y * img_h)), 5, (0, 255, 0), -1)
                #for shoulderRight
                showimg = cv2.circle(showimg, (int(self.received_pose.shoulderRight.x * img_w), int(self.received_pose.shoulderRight.y * img_h)), 5, (255, 0, 0), -1)
                #for shoulderLeft
                showimg = cv2.circle(showimg, (int(self.received_pose.shoulderLeft.x * img_w), int(self.received_pose.shoulderLeft.y * img_h)), 5, (0, 0, 255), -1)
                #for hipRight
                showimg = cv2.circle(showimg, (int(self.received_pose.hipRight.x * img_w), int(self.received_pose.hipRight.y * img_h)), 5, (255, 255, 0), -1)
                #for hipLeft
                showimg = cv2.circle(showimg, (int(self.received_pose.hipLeft.x * img_w), int(self.received_pose.hipLeft.y * img_h)), 5, (0, 255, 255), -1)
                #for pose_point in self.received_pose:
                #   cv2.circle(showimg, (int(pose_point.x), int(pose_point.y)), 5, (0, 255, 0), -1)

                # cutting image from chest to hips
                try:
                    if (self.received_pose.chest.y) < (self.received_pose.hipRight.y):
                        print("chest is higher than hip")
                        cut_y_up = int(self.received_pose.chest.y * img_h)
                        if (self.received_pose.hipRight.y) < 1:
                            print("hip is in image")
                            cut_y_down = int(self.received_pose.hipRight.y * img_h)
                        else:
                            cut_y_down = int(img_h)
                        cut_x_up = int(max(self.received_pose.shoulderRight.x, self.received_pose.shoulderLeft.x) * img_w)
                        cut_x_down = int(min(self.received_pose.shoulderRight.x, self.received_pose.shoulderLeft.x) * img_w)

                        # margin = 0.1
                        # cut_y_up -= int(cut_y_up * margin)
                        # cut_y_down += int(cut_y_down * margin)
                        # cut_x_up += int(cut_x_up * margin)
                        # cut_x_down -= int(cut_x_down * margin)



                        print(f"cut_y_up: {cut_y_up}, cut_y_down: {cut_y_down}, cut_x_up: {cut_x_up}, cut_x_down: {cut_x_down}")
                        #cut image from chest to hips
                        chestImg = showimg[cut_y_up:cut_y_down, cut_x_down:cut_x_up]
                        #contourImage = self.get_biggest_contour(chestImg)
                        
                        cv2.imshow('chestImg', chestImg)
                        #cv2.imshow('contourImage', contourImage)

                        #get mean color
                        mean_color = cv2.mean(chestImg)[:3]
                        mean_color = tuple(reversed(mean_color))
                        mean_color_rgb = cv2.cvtColor(np.array([[mean_color]], dtype=np.uint8), cv2.COLOR_BGR2RGB)[0][0]
                        print('Mean color of image foreground:', mean_color_rgb)
                        shirtColorstr = self.classifyColor(mean_color_rgb)
                        print('Shirt color is:', shirtColorstr)

                        color_rect = np.zeros((100, 100, 3), dtype=np.uint8)
                        color_rect[:, :] = mean_color_rgb
                        cv2.namedWindow('Color')
                        cv2.imshow('Color', color_rect)



                except Exception as e:
                    print(e)
                    print("Cannot cut image with current view")

                #cv2.imshow('chestImg', chestImg)
                cv2.imshow('image', showimg)
                cv2.waitKey(1)
                
            else:
                if self.received_image is None:
                    print("Waiting for image")
                if self.received_pose is None:
                    print("Waiting for pose")
            rospy.Rate(30).sleep()
                

if __name__ == '__main__':
    shirtColor().run()