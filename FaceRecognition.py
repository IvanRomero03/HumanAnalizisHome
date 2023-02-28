import cv2
from deepface.deepface import DeepFace
from matplotlib import pyplot as plt
import numpy as np
from typing import TypedDict, List
from util_types.recognition import FaceRecognitionRow

#find usa extract_faces en el background
face_results:List[FaceRecognitionRow] = DeepFace.find("mjFind.jpg", db_path = "faces", enforce_detection=False, detector_backend = 'retinaface')[0]
print(face_results)

# revisar https://stackoverflow.com/questions/70076895/deepface-for-generating-embedding-for-multiple-faces-in-single-image 
# para agregar rostros a la "base de datos"
