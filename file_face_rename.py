import os

# Path to folder with images
path = "detectorTest_db"

# List of images in folder
images = os.listdir(path)

# Renaming images
for count, image in enumerate(images):
    os.rename(f"{path}/{image}", f"{path}/face_{count}.jpg")


