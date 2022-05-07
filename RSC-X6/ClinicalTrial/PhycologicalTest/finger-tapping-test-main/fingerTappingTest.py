import cv2
import numpy as np
import mediapipe as mp
import time
import math


def fingertips(idNumber):
    lFinger = []
    if id == idNumber:
        lFinger.append(cx)
        lFinger.append(cy)
        pointList.append(lFinger)
        
def gradient(pt1, pt2):
    return (pt2[1]-pt1[1])/(pt2[0]-pt1[0])

def temp(ini, ang):
    tbs = []
    while True:
        end = time.time()
        temp = end - ini
        break
    tbs.append(temp)
    tbs.append(ang)
    return tbs
        
def getAngle(pointList):
    pt1,pt2,pt3 = pointList[-3:]
    # print(pt1,pt2,pt3)
    m1 = gradient(pt1, pt2)
    m2 = gradient(pt1, pt3)
    angR = math.atan((m2-m1)/(1+(m2*m1)))
    angD = abs(round(math.degrees(angR)))
    # print(angD)
    # cv2.putText(img, f'{int(angD)}', (pointList[0]), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0),2)
    return angD

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
ini = time.time()

tb = []

tempoExc = 0

while tempoExc<=1500:
    ret, img = cap.read()
    
    imgBGR = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgBGR)
    # print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks and tempoExc<=1500:
        for handLms in results.multi_hand_landmarks:
            pointList = []
            for id, lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx,cy)
                
                if id == 0 or id ==4 or id == 8:
                    cv2.circle(img, (cx,cy), 10, (0,0,255), cv2.FILLED)
                
                fingertips(0)
                fingertips(4)
                fingertips(8)

                pts = np.array(pointList, np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(img, [pts], True, (0,0,255), 3)
                
                if len(pointList) % 3 == 0 and len(pointList) !=0:
                    angD = getAngle(pointList)
                    cv2.putText(img, f'{int(angD)}', (pointList[0]), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0),2)
                    # t = temp(ini, angD) 
                    tb.append(temp(ini, angD))
                    tempoExc = tempoExc+1

            # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0),3)
    
    cv2.imshow('img', img)
    cv2.waitKey(1)
cv2.destroyAllWindows()

print(tb)