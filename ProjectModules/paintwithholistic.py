import mediapipe as mp
import cv2
import time
import os
import numpy as np

pTime = 0
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

folderPath = "menu"
myList = os.listdir(folderPath)
print(myList)

# overlayList = []
# for imPath in myList:
#     image = cv2.imread(f'{folderPath}/{imPath}')
#     overlayList.append(image)

overlayList = [cv2.imread(f'{folderPath}/{imPath}') for imPath in myList]

header = overlayList[0]

cap = cv2.VideoCapture(0)

################################
wCam, hCam = 1280, 720
################################

cap.set(3, wCam)
cap.set(4, hCam)

tipIds = [4, 8, 12, 16, 20]
drawColor = (255, 0, 255)
brushThickness = 15
xp, yp = 0, 0
imgCanvas = np.zeros((hCam, wCam, 3), np.uint8)


def set_color(color, t, r):  # color - BGR, thickness, circle_radius
    return mp_drawing.DrawingSpec(color=color, thickness=t, circle_radius=r)


def fingersUp(lmList, hand):
    fingers = []

    if hand == 0:  # left hand
        fingers.append(0)  # denoting left hand
        # Thumb
        if lmList[tipIds[0]].x > lmList[tipIds[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)
    elif hand == 1:  # right hand
        fingers.append(1)  # denoting right hand
        # Thumb
        if lmList[tipIds[0]].x > lmList[tipIds[0] - 1].x:
            fingers.append(0)
        else:
            fingers.append(1)

    # 4 Fingers
    for id in range(1, 5):
        if lmList[tipIds[id]].y < lmList[tipIds[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


def identify_pose(fingers, positions):
    global xp, yp
    # [0,0,1,1,0,0]
    temp = [0] * 6
    temp[0] = fingers[0]

    for i in range(6):
        for j in positions:
            if i == j:
                temp[i] = 1

    # print(temp, " and ", fingers)

    if fingers == temp:
        return True
    else:
        reset_gesture = [1] * 6
        reset_gesture[0] = fingers[0]

        if fingers == reset_gesture:
            xp, yp = 0, 0
        return False


def detection_start():
    global pTime, imgCanvas, header, drawColor, xp, yp, brushThickness

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                      set_color((80, 110, 10), 1, 1),
                                      set_color((80, 256, 121), 1, 1))
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      set_color((121, 22, 76), 2, 4),
                                      set_color((121, 44, 250), 2, 2))
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      set_color((121, 22, 76), 2, 4),
                                      set_color((121, 44, 250), 2, 2))
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      set_color((245, 117, 66), 2, 4),
                                      set_color((245, 66, 230), 2, 2))

            # Start Painting
            ## Right Hand
            if results.left_hand_landmarks is not None:
                rx1, ry1 = int(results.left_hand_landmarks.landmark[8].x * 1280), int(results.left_hand_landmarks.landmark[8].y * 700)
                rx2, ry2 = int(results.left_hand_landmarks.landmark[12].x * 1280), int(results.left_hand_landmarks.landmark[12].y * 700)

                rfingers = fingersUp(results.left_hand_landmarks.landmark, 1)

                # Selection Mode
                if identify_pose(rfingers, [2, 3]):
                    xp, yp = 0, 0
                    cv2.rectangle(image, (rx1, ry1 - 25), (rx2, ry2 + 25), drawColor, cv2.FILLED)
                    print("Selection Mode")
                    # Check for click
                    if ry1 < 125:
                        if 100 < rx1 < 200:
                            drawColor = (255, 0, 255)
                            header = overlayList[1]  # pencil tool
                        elif 250 < rx1 < 350:
                            drawColor = (255, 0, 255)
                            header = overlayList[2]  # Brush tool
                        elif 400 < rx1 < 510:
                            drawColor = (0, 0, 0)
                            header = overlayList[3]  # Shapes tool
                        # elif 550 < rx1 < 700:
                        #     drawColor = (255, 0, 255)
                        #     header = overlayList[4]  # Shapes tool
                        elif 750 < rx1 < 850:
                            drawColor = (255, 0, 255)
                            header = overlayList[4]  # Undo tool
                        elif 900 < rx1 < 1000:
                            drawColor = (255, 0, 255)
                            header = overlayList[5]  # Redo tool
                        elif 1100 < rx1 < 1200:
                            drawColor = (255, 0, 255)
                            imgCanvas = np.zeros((hCam, wCam, 3), np.uint8)
                            header = overlayList[6]  # Delete tool
                            # header = overlayList[0]
                        # else:
                        #     header = overlayList[0]

                # Canvas Mode
                if identify_pose(rfingers, [2]):
                    cv2.circle(image, (rx1, ry1), 15, drawColor, cv2.FILLED)
                    print("Drawing Mode")

                    if xp == 0 and yp == 0:
                        xp, yp = rx1, ry1

                    # if drawColor == (0, 0, 0):
                    #     brushThickness = 50
                    # else:
                    drawColor = (255, 0, 255)
                    brushThickness = 15

                    cv2.line(image, (xp, yp), (rx1, ry1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (rx1, ry1), drawColor, brushThickness)

                    xp, yp = rx1, ry1

                # erase mode
                if identify_pose(rfingers, [2, 3, 4]):
                    drawColor = (0, 0, 0)

                    if xp == 0 and yp == 0:
                        xp, yp = rx1, ry1

                    brushThickness = 50

                    cv2.line(image, (xp, yp), (rx1, ry1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (rx1, ry1), drawColor, brushThickness)

                    xp, yp = rx1, ry1



            # FPS CODE
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(image, f'FPS: {int(fps)}', (40, 250), cv2.FONT_HERSHEY_COMPLEX,
                        1, (255, 0, 0), 3)

            # Combining with image
            imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            image = cv2.bitwise_and(image, imgInv)
            image = cv2.bitwise_or(image, imgCanvas)

            # image = cv2.addWeighted(image, 0.5, imgCanvas, 0.5, 0)  # blend with the canvas

            # Setting the header image
            image[0:125, 0:1280] = header
            cv2.imshow("Img", image)
            cv2.waitKey(1)


if __name__ == '__main__':
    detection_start()
