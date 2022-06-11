import cv2
import numpy as np
import time
import os
import handdetectionmodule as htm

folderPath = "menu"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]

cap = cv2.VideoCapture(0)

################################
wCam, hCam = 1280, 720
################################

cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(detectionCon=0.85)

drawColor = (255, 0, 255)
brushThickness = 15
xp, yp = 0, 0
imgCanvas = np.zeros((hCam, wCam, 3), np.uint8)
while True:
    # 1. Import image
    success, image = cap.read()
    image = cv2.flip(image, 1)

    # 2. Find Hand Landmarks
    image = detector.findHands(image)
    lmlist = detector.findPosition(image, draw=False)

    if len(lmlist) != 0:
        # tip of index and middle fingers
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # 4. If Selection mode - two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print(x1, y1)
            cv2.rectangle(image, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
            print("Selection Mode")
            # Check for click
            if y1 < 125:
                if 100 < x1 < 200:
                    drawColor = (255, 0, 255)
                    header = overlayList[1]  # pencil tool
                elif 250 < x1 < 350:
                    drawColor = (255, 0, 255)
                    header = overlayList[2]  # Brush tool
                elif 400 < x1 < 510:
                    drawColor = (0, 0, 0)
                    header = overlayList[3]  # Eraser tool
                elif 550 < x1 < 700:
                    drawColor = (255, 0, 255)
                    header = overlayList[4]  # Shapes tool
                elif 750 < x1 < 850:
                    drawColor = (255, 0, 255)
                    header = overlayList[5]  # Undo tool
                elif 900 < x1 < 1000:
                    drawColor = (255, 0, 255)
                    header = overlayList[6]  # Redo tool
                elif 1100 < x1 < 1200:
                    drawColor = (255, 0, 255)
                    imgCanvas = np.zeros((hCam, wCam, 3), np.uint8)
                    header = overlayList[0]  # Delete tool
                    # header = overlayList[0]
                # else:
                #     header = overlayList[0]

        # 5. If Drawing mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(image, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                brushThickness = 50
            else:
                brushThickness = 15

            cv2.line(image, (xp, yp), (x1, y1), drawColor, brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    # Combining with image
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    image = cv2.bitwise_and(image, imgInv)
    image = cv2.bitwise_or(image, imgCanvas)

    # Setting the header image
    image[0:125, 0:1280] = header
    # image = cv2.addWeighted(image, 0.5, imgCanvas, 0.5, 0)  # blend with the canvas
    cv2.imshow("image", image)
    # cv2.imshow("Canvas", imgInv)
    cv2.waitKey(1)
