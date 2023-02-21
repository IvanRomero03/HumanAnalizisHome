import cv2
from deepface import DeepFace


attributes_results = DeepFace.analyze(img_path = "cara1.jpg", actions = ['age', 'gender', 'race'])
print(attributes_results)
