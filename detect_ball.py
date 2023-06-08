import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
import math

cap = cv2.VideoCapture(r"D:\Desktop\智慧城市期末project\shot_predict\mask_video\V1.mp4")

myColorFinder = ColorFinder(True)
hsvVals = 'orange'


while True:
    #success, img = cap.read()
    #img = img[:,200:700]
    img = cv2.imread(r".\shot_predict\mask_video\shoot.jpg")
    
    imgColor, mask = myColorFinder.update(img, hsvVals)


    imgColor = cv2.resize(imgColor, (0, 0), None, 0.7, 0.7)
    cv2.imshow("imagecolor", imgColor)
    cv2.waitKey(50)