#!/usr/bin/env python3

import sys
import rospy
from humanAnalyzer.srv import faces_info, faces_infoResponse



def json_client():
    rospy.wait_for_service('onCamFacesService')
    try:
        jsonsrv = rospy.ServiceProxy('onCamFacesService', faces_info)
        resp = jsonsrv()
        return resp.faces_info
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def usage():
    return "%s [color]"%sys.argv[0]

if __name__ == "__main__":
    print("Requesting ...")
    print(json_client())
    