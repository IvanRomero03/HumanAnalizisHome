import os
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from dotenv import load_dotenv
import requests
import uuid
from PIL import Image

class AzureFaceRecognizer:
    def __init__(self, endpoint, subscription_key):
        self.face_client = FaceClient(endpoint, CognitiveServicesCredentials(subscription_key))

    def addFace(self, group_id, person_name, image_url, person):
        #image_data = requests.get(image_url).content
        # response = self.face_client.person_group_person.create(group_id, person_name)
        person_id = str(uuid.uuid4())
        # person_id = response.person_id
        #self.face_client.person_group_person.add_face_from_stream(group_id, person_id, image_data)
        detected_faces = self.face_client.face.detect_with_url(url=image_url, detection_model='detection_03', recognition_model='recognition_04', return_face_attributes=['qualityForRecognition'])
        if detected_faces:
            for face in detected_faces:
                if face.face_attributes.quality_for_recognition != "High":
                    print("Face quality is low: {}".format(face.face_attributes.quality_for_recognition))
                    return
                else:
                    print("Face quality is high: {}".format(face.face_attributes.quality_for_recognition))
                    self.face_client.person_group_person.add_face_from_url(person_group_id=group_id,person_name=person_name,url=image_url, person_id=face.person_id)
        else:
            print("No face detected")
            return
        #self.face_client.person_group_person.add_face_from_url(person_group_id=group_id,person_name=person_name, url=image_url)

    def testFace(self, group_id, image_url):
        image_data = requests.get(image_url).content
        faces = self.face_client.face.detect_with_stream(image_data)
        face_ids = [face.face_id for face in faces]
        response = self.face_client.face.identify(face_ids, group_id)
        candidates = response[0].candidates
        if candidates:
            person_id = candidates[0].person_id
            person = self.face_client.person_group_person.get(group_id, person_id)
            return person.name
        else:
            return "Unknown"

if __name__ == '__main__':
    load_dotenv()
    subscription_key = os.getenv('AZURE_KEY1')
    endpoint = os.getenv('AZURE_ENDPOINT')
    group_id = str(uuid.uuid4())
    print(group_id)

    recognizer = AzureFaceRecognizer(endpoint, subscription_key)

    recognizer.face_client.person_group.create(person_group_id=group_id, name=group_id, recognition_model='recognition_04')

    #recognizer.face_client.person_group_person.create(person_group_id=group_id, name=group_id, recognition_model='recognition_04')

    # Adding the starting group
    recognizer.addFace(group_id, "Jeff", "https://bfmvwivyerrefrhrlmxx.supabase.co/storage/v1/object/public/imagenes/Home/jeff.jpg")
    recognizer.addFace(group_id, "Michael Jordan", "https://bfmvwivyerrefrhrlmxx.supabase.co/storage/v1/object/public/imagenes/Home/michael-jordan.jpg")

    # Adding a new face
    recognizer.addFace(group_id, "Schum", "https://bfmvwivyerrefrhrlmxx.supabase.co/storage/v1/object/public/imagenes/Home/schum.jpg")

    # Testing a new face
    result = recognizer.testFace(group_id, "https://bfmvwivyerrefrhrlmxx.supabase.co/storage/v1/object/public/imagenes/Home/rand.jpg")
    print(result)  # Output: Unknown
