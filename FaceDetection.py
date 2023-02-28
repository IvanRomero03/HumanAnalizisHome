import cv2
from deepface.deepface import DeepFace
from matplotlib import pyplot as plt
import numpy as np

#print(DeepFace.analyze(img_path = "arbol.jpg", actions = ['emotion']))

#face_objs = DeepFace.extract_faces("cara1.jpg", detector_backend = 'retinaface')

face_objs = DeepFace.extract_faces("cara1.jpg", detector_backend = 'retinaface')

# get face imgs in region within face_objs
print(face_objs[0].keys())
face_imgs = []
for face_obj in face_objs:
    face_area = face_obj["facial_area"]
    print(face_area)
    #{'x': 201, 'y': 120, 'w': 352, 'h': 352}
    with open("cara1.jpg", "rb") as f:
        img = f.read()
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    face_img = img[face_area["y"]:face_area["y"]+face_area["h"], face_area["x"]:face_area["x"]+face_area["w"]]
    face_imgs.append(face_img)

# plot face imgs
for i in range(len(face_imgs)):
    plt.subplot(1, len(face_imgs), i+1)
    plt.imshow(face_imgs[i][:, :, ::-1])
    plt.axis('off')
plt.show()

# plot face objs
for i in range(len(face_objs)):
    plt.subplot(1, len(face_objs), i+1)
    plt.imshow(face_objs[i]["face"])
    plt.axis('off')
plt.show()
