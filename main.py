import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

#variable
width, height = 1200, 720
folderPath = "Presentation"

#camera setup
capture = cv2.VideoCapture(0)
capture.set(3, width)
capture.set(4, height)

#get the list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)

#Variables
imgNumber = 0
hs, ws = int(120*1), int(213*1)
gestureThreshold = 500
buttonPressed = False
buttonCount = 0
buttonDelay = 10
annotations = [[]]
annotationNumber = 0
annotationStart = False

#Hand Detection
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    success, img = capture.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)
    #Adding webcam image on the slides
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w-ws:w] = imgSmall

    hands, img = detector.findHands(img)
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

    if hands and buttonPressed is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        lmList = hand['lmList']

        # Contrain values for easier drawing(so that we need not move our hand to a great extent)
        indexFinger = lmList[8][0], lmList[8][1]
        x_val = int(np.interp(lmList[8][0], [width//2, w], [0, width]))
        y_val = int(np.interp(lmList[8][1], [150, height-150], [0, height]))
        indexFinger = x_val, y_val

        if cy<= gestureThreshold: # if hand is at the height of the face
            #Gesture 1 - left
            if fingers == [1, 0, 0, 0, 0]:
                annotationStart = False
                print("Left")
                if imgNumber>0:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = 0
                    imgNumber -= 1

            #Gesture 2 - right
            if fingers == [0, 0, 0, 0, 1]:
                annotationStart = False
                print("Right")
                buttonPressed = True
                if imgNumber < len(pathImages)-1:
                    annotations = [[]]
                    annotationNumber = 0
                    imgNumber += 1

        #Gesture 3 - Show pointer
        if fingers == [0, 1, 0, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotationStart = False

        #Gesture 4 - Drawing
        if fingers == [0, 1, 1, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotations[annotationNumber].append(indexFinger)
        else:
            annotationStart = False

        #Gesture 5 - Eraser
        if fingers == [1, 1, 1, 1, 1]:
            if annotations:
                if annotationNumber >= 0:
                    annotations.pop(-1)
                    annotationNumber -= 1
                    buttonPressed = True


    else:
        annotationStart = False

    #Button press iterations
    if buttonPressed:
        buttonCount += 1
        if buttonCount > buttonDelay:
            buttonCount = 0
            buttonPressed = False

    for i in range(len(annotations)):
        for j in range(len(annotations[i])):
            if j != 0:
               cv2.line(imgCurrent, annotations[i][j-1], annotations[i][j], (0, 0, 200), 12)




    cv2.imshow("Image", img)#Video capture from device camera
    cv2.imshow("Slides", imgCurrent)#Slide from device storage
    key = cv2.waitKey(1)
    if key == ord('q'):
        break