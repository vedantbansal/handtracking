import cv2 as cv
import HandTrackingModule as htm
import numpy as np
import sys
import matplotlib.pyplot as plt

#Open camera
cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85, trackCon=0.7)
img = 1 #Colour number or image number

# header = cv.imread(cv.samples.findFile(f"PaintHeaders\\{img}.png")) # Top image

#Check is camera is open
if not cap.isOpened():
    print("Camera can not be opened. Exiting...")
    sys.exit()

canvas = np.zeros((720,1280,3), np.uint8)
xP, yP = 0, 0

def getColour(img):
    if img == 1:
        colour = (0,0,255)
    
    elif img == 2:
        colour = (255,0,0)

    elif img == 3:
        colour = (0,255,0)

    elif img == 4:
        colour = (0,255,255)

    elif img == 5:
        colour = (0,110,200)

    elif img == 6:
        colour = (255,0,255)
    
    elif img == 7:
        colour = (0,0,0)

    return colour

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not recieved. Exiting...")
        sys.exit()

    frame = cv.flip(frame, 1)
    frame = detector.findHands(frame) #detect hand in frame
    coordinates = detector.findPosition(frame)

    #selection mode (2 fingers raised)
    if detector.isFingerOpen(frame, 1) and detector.isFingerOpen(frame, 2):
        xP, yP = 0, 0
        _, x1, y1 = coordinates[8] #index finger tip coordinates
        _, x2, y2 = coordinates[12] #middle finger tip coordinates
        mid = ((x2+x1)//2, (y1+y2)//2)

        cv.circle(frame, mid, 20, getColour(img), -1)
        
        #find the colour selected using middle point located
        if 15 <= mid[1] <=105:
            for i in range(0,6):
                if (50+ i *(180)) <= mid[0] <=  (130+ i *(180)):
                    img = i+1
            
            if 1137 <= mid[0] <= 1228:
                img = 7
        
        #change header image
        # header = cv.imread(cv.samples.findFile(f"PaintHeaders\\{img}.png"))
        
    #draw mode (1 finger raised)
    if detector.isFingerOpen(frame, 1) and detector.isFingerOpen(frame, 2) == False:
        _, xN, yN = coordinates[8] #index finger tip coordinates
        if xP == 0 and yP == 0 :
            xP, yP = xN, yN
        
        cv.circle(frame, (xN, yN), 15, getColour(img), -1)
        cv.line(canvas, (xP, yP), (xN, yN), getColour(img), 20)
        xP, yP = xN, yN

    #display the drawing in real time
    canvasGray = cv.cvtColor(canvas, cv.COLOR_BGR2GRAY)
    _, maskInv = cv.threshold(canvasGray, 50, 255, cv.THRESH_BINARY_INV)
    maskInv = cv.cvtColor(maskInv, cv.COLOR_GRAY2BGR)
    frame = cv.bitwise_and(frame, maskInv)
    frame = cv.bitwise_or(frame, canvas)
    
    #add the header image
    top = frame[0:120, 0:1280]
    # top = cv.addWeighted(header, 1, top, 0.3, 1)
    frame[0:120, 0:1280] = top

    cv.imshow("Paint", frame)
    
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()