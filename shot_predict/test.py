import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np 
import math


#initialize the video
cap = cv2.VideoCapture(r"D:\Desktop\智慧城市期末project\shot_predict\Videos\vid (4).mp4")

#create the color finder object
myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 0, 'smin': 138, 'vmin': 0, 'hmax': 156, 'smax': 255, 'vmax': 255}

#Variables
posListX, posListY = [], []
xList = [item for item in range(0,1300)] 
prediciton = False

while True:
    success, img = cap.read()
    img = img[0:900, :]
    #img = cv2.imread(r".\shot_predict\Ball.png")
    
 
    #Find the color Ball
    imgColor, mask = myColorFinder.update(img, hsvVals)

    #Find the location of the ball
    imgContours, contours = cvzone.findContours(img, mask,minArea=500)
    
    #display points frame by frame
    if contours:
        posListX.append(contours[0]['center'][0])
        posListY.append(contours[0]['center'][1])
        
        
    #polynomial regression y = AX^2 + BX + C
    #find te coefficients
    if posListX:
        A, B, C = np.polyfit(posListX,posListY,2)
       
        for i, (posX,posY) in enumerate(zip(posListX,posListY)):
            pos = (posX,posY)
            cv2.circle(imgContours, pos,10, (0,255,0),cv2.FILLED)
            if i ==0:
                cv2.line(imgContours, pos, pos, (0,255,0),2)
            else:
                cv2.line(imgContours, pos, (posListX[i-1],posListY[i-1]), (0,255,0),10)


        
        for x in xList:
            y = int(A*x**2 + B*x + C) #draw polynomial with the coeff
            
            
            cv2.circle(imgContours, (x,y),2, (255,0,255),cv2.FILLED)

        #use ten frames to predict the basket
        if (len(posListX)) < 10:
                    
            #prediction
            #xvalues 330 to 430 , yvalues 590
            a = A
            b = B
            c = C-590 
            #what value is x when y is equal to 590
            x = (-b - math.sqrt(b**2 - (4*a*c)))/(2*a)
            prediciton = 330 < x <430
            
        if prediciton:
            cvzone.putTextRect(imgContours, "Basket", (50,150), scale = 5, 
                                thickness =5, colorR = (0,200,0), offset = 20)
                
        else:
            cvzone.putTextRect(imgContours, "No Basket", (50,150), scale = 5, 
                                thickness =5, colorR = (0,0,200), offset = 20)
        

    imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7)
    #cv2.imshow("Image", img)
    cv2.imshow("ImageColor", imgContours)
    cv2.waitKey(1)
