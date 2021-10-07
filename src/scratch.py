import numpy as np


import cv2 as cv
import os

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


filename = "vid1.mp4"
path = find(filename, ".\\")
colorType = cv.COLOR_RGB2BGR

frametime = 10
vid = cv.VideoCapture(path)
ret, img = vid.read();
cv.imshow("firstFrame", img)

while True:
    ret, img = vid.read();
    
    gray = cv.cvtColor(img, colorType)
    cv.imshow("firstFrame", gray)

    if cv.waitKey(frametime) & 0xFF == ord('q'):
        break

