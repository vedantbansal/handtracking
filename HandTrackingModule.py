from typing import Coroutine
import cv2 as cv
import mediapipe as mp
import time
import sys
import math

class handDetector():

    def __init__(self, mode=False, maxHands=2, detect_conf=0.5, track_conf = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detect_conf = detect_conf
        self.track_conf = track_conf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detect_conf, self.track_conf)
        self.mpDraw = mp.solutions.drawing_utils
        self.pTime = 0

    # finding the slope between two points
    def __slope(self, x1, y1, x2, y2):
        try:
            m = (y2 - y1)/(x2 - x1)

        except ZeroDivisionError:
            m = 1.633123935319537e+16
        
        return m

    # finding angle between two lines
    def __angle(self, m1, m2):
        try:
            ang = math.atan((m2-m1)/(1+m2*m1))

        except ZeroDivisionError:    
            ang = math.atan(1.633123935319537e+16)
        
        return ang

    # detect hands in the coming frames
    def findHands(self, frame, draw=True):
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)

        return frame

    # returns position of all hand landmarks
    def findPosition(self, frame, handNo=0, draw=False):
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                height, length, _ = frame.shape
                cX, cY = int(lm.x * length), int(lm.y * height)
                lmList.append([id, cX, cY])

                if draw:
                    cv.circle(frame, (cX, cY), 15, (255, 0, 255), cv.FILLED)

        return lmList

    # Detect if given finger is open
    def isFingerOpen(self, frame, finger):
        lmList = self.findPosition(frame)
        if lmList:
            _, x1, y1 = lmList[1+finger*4]
            _, x2, y2 = lmList[2+finger*4]
            _, x3, y3 = lmList[3+finger*4]
            _, x4, y4 = lmList[4+finger*4]
            
            m1 = self.__slope(x1, y1, x2, y2)
            m2 = self.__slope(x2, y2, x3, y3)
            m3 = self.__slope(x3, y3, x4, y4)
            
            ang2 = self.__angle(m1, m2)
            ang3 = self.__angle(m2, m3)

            if -0.15<= ang2 <= 0.15 and -0.15 <= ang3 <= 0.15:
                return True
            else:
                return False

    # Detect if thumb is open
    def isThumbOpen(self, frame):
        lmList = self.findPosition(frame)
        if lmList:
            _, x1, y1 = lmList[0]
            _, x2, y2 = lmList[1]
            _, x3, y3 = lmList[2]

            m1 = self.__slope(x1, y1, x2, y2)
            m2 = self.__slope(x2, y2, x3, y3)

            ang2 = self.__angle(m1, m2)

            if -0.32<= ang2 <= 0.32:
                return True
            else:
                return False
    
    # returns fps of coming stream
    def getFPS(self):
        self.cTime = time.time()
        fps = 1/(self.cTime - self.pTime)
        self.pTime = self.cTime

        return int(fps)

def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = handDetector( detect_conf=0.85)

    if not cap.isOpened():
        print("Camera can not be opened")
        sys.exit()

    while True:
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)

        frame = detector.findHands(frame)
        lmlist = detector.findPosition(frame, draw=False)
        fin = [detector.isFingerOpen(frame, 1), detector.isFingerOpen(frame, 2), 
                detector.isFingerOpen(frame, 3), detector.isFingerOpen(frame, 4),
                detector.isThumbOpen(frame)]
        fps = detector.getFPS()
        print(fin)
        #print(fps)
        if not ret:
            print("Can't recieve frame. Exiting...")

        cv.imshow("Frame Output", frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()