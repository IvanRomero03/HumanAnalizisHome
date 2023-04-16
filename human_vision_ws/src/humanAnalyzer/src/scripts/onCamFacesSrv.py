#!/usr/bin/env python3
import rospy
import rospkg
import json
from humanAnalyzer.msg import face, face_array, face_info
from humanAnalyzer.srv import faces_info, faces_infoResponse

class onCamFacesSrv:

    def __init__(self):
        rospy.init_node('onCamFacesSrv')
        self.facesmsg = []
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('humanAnalyzer')
        self.json_path = f"{pkg_path}/src/scripts/identities.json"
        self.faceSub = rospy.Subscriber(
            'faces', face_array, self.listener, queue_size=10)
        self.jsonSrv = rospy.Service('onCamFacesService', faces_info, self.handle_request)

    def listener(self, data):
        self.facesmsg = data.faces
        print(data)
    
    def printReceived(self):
        print(self.facesmsg)
    
    #-----------SERVER---------------------------------
    def handle_request(self, req):
        print("Received request")
        facesResponse = faces_infoResponse()
        for facemsg in self.facesmsg:
            # Generating coordinates in (x,y) format
            faceResponse = face_info()
            faceResponse.identity = facemsg.identity
            faceResponse.x = facemsg.x
            faceResponse.y = facemsg.y
            faceResponse.w = facemsg.w
            faceResponse.h = facemsg.h
            #getting attributes and writing to json
            #load json
            data = {}
            try:
                f = open(self.json_path)
                data = json.load(f)
                face_id = facemsg.identity
                faceResponse.age = data[face_id]["age"]
                faceResponse.gender = data[face_id]["gender"]
                faceResponse.race = data[face_id]["race"]
                
            except Exception as e:
                print("JSON file uncomplete or unexistent")
                print(e)
            facesResponse.faces_info.append(faceResponse)
        print(facesResponse)
        return facesResponse
        

    def run(self):
        
        print("Ready to receive requests")
        rospy.spin()

onCamFacesSrv().run()
