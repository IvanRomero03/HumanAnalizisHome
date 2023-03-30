import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO

from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials

from dotenv import load_dotenv

class AzureFaceList:
    def __init__(self, face_list_id, azure_key, azure_endpoint):
        self.face_list_id = face_list_id
        self.face_client = FaceClient(azure_endpoint, CognitiveServicesCredentials(azure_key))
        self.face_client.face_list.create(face_list_id, name = face_list_id)

    def addFace(self, single_face_image_url):
        response = self.face_client.face_list.add_face_from_url(self.face_list_id, single_face_image_url, user_data=None, target_face=None, detection_model='detection_01',
                    custom_headers=None, raw=False)
        print(response.persisted_face_id)
        return response.persisted_face_id

    def deleteFace(self, persisted_face_id):
        self.face_client.face_list.delete_face(self.face_list_id, persisted_face_id, custom_headers=None, raw=False)
    
    def getFaceList(self):
        print(self.face_client.face_list.get(self.face_list_id))
        return self.face_client.face_list.get(self.face_list_id)
    
    def verifyFace(self, face_id):
        candidates = self.face_client.face.find_similar(face_id, self.face_list_id, max_num_of_candidates_returned=1)
        for candidate in candidates:
            print(candidate)
            # if candidate.confidence > 0.5:
            #     return candidate
            # else:
            #     return None
            if candidate.confidence > 0.5:
                return candidate
            else:
                return None
            
    def verifyFaceFromUrl(self, single_face_image_url):
        detected_faces = self.face_client.face.detect_with_url(single_face_image_url, detectionModel='detection_02')
        if not detected_faces:
            raise Exception('No face detected from image {}'.format(single_face_image_url))
        face_id = detected_faces[0].face_id
        candidates = self.face_client.face.find_similar(face_id, self.face_list_id, max_num_of_candidates_returned=1)
        for candidate in candidates:
            print(candidate)
            if candidate.confidence > 0.5:
                return candidate
            else:
                return None
    
    # def detectFace(self, single_image_name):
    #     detected_faces = self.face_client.face.detect_with_stream(image=io.BytesIO(single_image_name), detectionModel='detection_02')
    #     if not detected_faces:
    #         raise Exception('No face detected from image {}'.format(single_image_name))
    #     return detected_faces
    
    def deleteFaceList(self):
        self.face_client.face_list.delete(self.face_list_id, custom_headers=None, raw=False)

    # def __del__(self):
    #     self.deleteFaceList()

if __name__ == "__main__":
    load_dotenv()
    AZURE_KEY=os.getenv("AZURE_KEY1")
    AZURE_ENDPOINT=os.getenv("AZURE_ENDPOINT")
    face_list_id = "face_list_id3"
    azureFaceList = AzureFaceList(face_list_id, AZURE_KEY, AZURE_ENDPOINT)
    # https://bfmvwivyerrefrhrlmxx.supabase.co/storage/v1/object/public/imagenes/Home/jeff.jpg
    # https://bfmvwivyerrefrhrlmxx.supabase.co/storage/v1/object/public/imagenes/Home/michael-jordan.jpg
    # https://bfmvwivyerrefrhrlmxx.supabase.co/storage/v1/object/public/imagenes/Home/schum.jpg
    # https://bfmvwivyerrefrhrlmxx.supabase.co/storage/v1/object/public/imagenes/Home/rand.jpg

    # Add first two faces to the face list
    persisted_face_id1 = azureFaceList.addFace("https://bfmvwivyerrefrhrlmxx.supabase.co/storage/v1/object/public/imagenes/Home/jeff.jpg")
    persisted_face_id2 = azureFaceList.addFace("https://bfmvwivyerrefrhrlmxx.supabase.co/storage/v1/object/public/imagenes/Home/michael-jordan.jpg")

    # Check if face 3 is in the face list
    print("Verifying face 3...")
    verify_result3 = azureFaceList.verifyFaceFromUrl("https://bfmvwivyerrefrhrlmxx.supabase.co/storage/v1/object/public/imagenes/Home/jef2.jpg")
    print(verify_result3)
    if verify_result3 is None:
        # Add face 3 to the face list
        print("Face 3 is not in the face list")
        persisted_face_id3 = azureFaceList.addFace("https://bfmvwivyerrefrhrlmxx.supabase.co/storage/v1/object/public/imagenes/Home/schum.jpg")
    else:
        print("Face 3 is already in the face list")

    # Same with face 4
    print("Verifying face 4...")
    verify_result4 = azureFaceList.verifyFaceFromUrl("https://bfmvwivyerrefrhrlmxx.supabase.co/storage/v1/object/public/imagenes/Home/rand.jpg")
    print(verify_result4)
    if verify_result4 is None:
        # Add face 4 to the face list
        print("Face 4 is not in the face list")
        persisted_face_id4 = azureFaceList.addFace("https://bfmvwivyerrefrhrlmxx.supabase.co/storage/v1/object/public/imagenes/Home/rand.jpg")
    else:
        print("Face 4 is already in the face list")
    
    # Print list of faces in the face list
    print("Faces in the face list:")
    azureFaceList.getFaceList()
    azureFaceList.deleteFaceList()


                

