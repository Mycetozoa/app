import os
import sys
from PIL import Image


dirname = '/home/kuznech/Work/furniture/classifier/data/'
cnt=0
for folder, dirs, files in os.walk(dirname):
    for file in files:
        try:
            img_path = os.path.join(folder, file)
            img=Image.open(img_path)
        except OSError:
            print("FILE: ", img_path, "is corrupt!")
            cnt+=1
            os.remove(img_path)
print("Successfully Completed Operation! Files Courrupted are ", cnt)