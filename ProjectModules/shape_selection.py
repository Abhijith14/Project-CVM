import math
import os

import mediapipe as mp
import cv2
import time
import numpy as np

pTime = 0
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

################################
# Cam Variables
wCam, hCam = 1280, 720
cap.set(3, wCam)
cap.set(4, hCam)
################################


###############################
# opencv variables
tipIds = [4, 8, 12, 16, 20]
###############################


################################
# Color Variables
drawColor = (255, 0, 255)
col1, col2, col3, col4 = (243, 179, 147), (134, 144, 228), (117, 215, 247), (153, 199, 145)
tc1, tc2, tc3, tc4 = col1, col2, col3, col4
selectionCol = (0, 0, 255)
################################


################################
# Drawing Variables
brushThickness = 15
xp, yp = 0, 0
imgCanvas = np.zeros((hCam, wCam, 3), np.uint8)
################################


###############################
# Slider Vatiables
percent = 20

compare = False
##############################


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


def add_landmarks(frame, holistic):
    global mp_drawing, mp_holistic

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

    return image, results


def add_fps(image):
    global pTime
    # FPS CODE
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(image, f'FPS: {int(fps)}', (40, 250), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)
    return image


def add_desc(image, selectionCol, heading, s1, s2, s3=''):
    cv2.putText(image, heading, (800, 200), cv2.FONT_HERSHEY_COMPLEX, 1, selectionCol, 2)
    cv2.putText(image, s1, (800, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                selectionCol, 1)
    cv2.putText(image, s2, (800, 280), cv2.FONT_HERSHEY_SIMPLEX, 1,
                selectionCol, 1)
    if s3 != '':
        cv2.putText(image, s3, (800, 320), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    selectionCol, 1)

    return image


def get_coordinates(landmark, index):
    return int(landmark[index].x * 1280), int(landmark[index].y * 700)


def get_distance(x1, y1, x2, y2):
    return int(math.hypot(x2 - x1, y2 - y1))


def create_menuWheel(image, landmark):
    rx1, ry1 = get_coordinates(landmark, 0)
    rx2, ry2 = get_coordinates(landmark, 8)
    rx3, ry3 = get_coordinates(landmark, 12)

    d1 = get_distance(rx1, ry1, rx2, ry2) + 50

    cv2.circle(image, (rx1 - 50, (ry1 - d1) + 20), 12, col1, cv2.FILLED)
    cv2.circle(image, (rx1, (ry1 - d1) + 10), 12, col2, cv2.FILLED)
    cv2.circle(image, (rx1 + 50, (ry1 - d1) + 10), 12, col3, cv2.FILLED)
    cv2.circle(image, (rx1 + 100, (ry1 - d1) + 30), 12, col4, cv2.FILLED)

    cv2.circle(image, (rx3, ry3 - 25), 17, selectionCol)

    return image


def detection_start():
    global pTime, drawColor, col1, col2, col3, col4, tc1, tc2, tc3, tc4, selectionCol, mp_holistic
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frame.flags.writeable = False

            image, results = add_landmarks(frame, holistic)

            image = add_fps(image)

            # Start Choosing
            ## Right Hand
            if results.left_hand_landmarks is not None:

                rfingers = fingersUp(results.left_hand_landmarks.landmark, 1)
                # print(rfingers)

                if results.right_hand_landmarks is not None:
                    lfingers = fingersUp(results.right_hand_landmarks.landmark, 0)

                    if rfingers[1:] == [0, 1, 1, 1, 1] and lfingers[2:] == [0, 0, 0, 0]:

                        rx1, ry1 = get_coordinates(results.left_hand_landmarks.landmark, 0)
                        rx2, ry2 = get_coordinates(results.left_hand_landmarks.landmark, 8)
                        rx3, ry3 = get_coordinates(results.left_hand_landmarks.landmark, 12)

                        d2 = get_distance(rx2, ry2, rx3, ry3)
                        d3 = get_distance(rx1, ry1, rx3, ry3)

                        if d3 < 350:

                            image = create_menuWheel(image, results.left_hand_landmarks.landmark)
                            # print(d2, d3)

                            # todo: Creating a hand based proper wheel alignment

                            if abs(rx3 - (rx1 - 50)) < 20:
                                selectionCol = (255, 0, 0)
                                col1, col2, col3, col4 = selectionCol, tc2, tc3, tc4

                                image = add_desc(image, selectionCol,
                                                 heading='Normal Mode',
                                                 s1='A mode in which no special',
                                                 s2='gestures are used apart from',
                                                 s3='the menu gesture.')

                            elif abs(rx3 - rx1) < 20:
                                selectionCol = (0, 0, 255)
                                col1, col2, col3, col4 = tc1, selectionCol, tc3, tc4

                                image = add_desc(image, selectionCol,
                                                 heading='Drawing Mode',
                                                 s1='A mode for drawing and',
                                                 s2='writing, creating shapes and',
                                                 s3='much more.')

                                # if d2 < 30:
                                #     print("Drawing Mode")
                                #     # countdown of 5
                                #     while True:
                                #         status = drawing_mode()
                                #         if status:
                                #             break
                                drawing_mode()

                            elif abs(rx3 - (rx1 + 50)) < 20:
                                selectionCol = (0, 255, 255)
                                col1, col2, col3, col4 = tc1, tc2, selectionCol, tc4

                                image = add_desc(image, selectionCol,
                                                 heading='Canvas Mode',
                                                 s1='A mode for playing around',
                                                 s2='with elements, creating',
                                                 s3='new screens.')

                            elif abs(rx3 - (rx1 + 100)) < 20:
                                selectionCol = (0, 255, 0)
                                col1, col2, col3, col4 = tc1, tc2, tc3, selectionCol

                                image = add_desc(image, selectionCol,
                                                 heading='Demo Mode',
                                                 s1='A mode for working with',
                                                 s2='images and videos.')

                            else:
                                col1, col2, col3, col4 = tc1, tc2, tc3, tc4

                            ## Keep a point constant
                            # try:
                            #     cv2.circle(image, (crx1, cry1-25), 15, drawColor, cv2.FILLED)
                            # except UnboundLocalError:
                            #     crx1, cry1 = tuple((rx1, ry1))

            cv2.imshow("Canvas", image)
            cv2.waitKey(1)


def drawing_mode():
    global pTime, drawColor, col1, col2, col3, col4, tc1, tc2, tc3, tc4, selectionCol, mp_holistic
    global xp, yp, brushThickness, imgCanvas, compare

    header, overlayList = get_menu()

    with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frame.flags.writeable = False

            image, results = add_landmarks(frame, holistic)

            image = add_fps(image)

            # Start Choosing

            # Reset
            if (header == overlayList[6]).all():
                header = overlayList[0]

            # if checkHandexists(results, 0):
            #     lfingers = fingersUp(results.right_hand_landmarks.landmark, 0)
            #     if lfingers[2:] == [0, 0, 0, 0]:
            #         header = overlayList[0]

            ## Left Hand
            # Selection Using Left
            if results.right_hand_landmarks is not None:
                xp, yp = 0, 0
                rx6, ry6 = get_coordinates(results.right_hand_landmarks.landmark, 8)
                rx7, ry7 = get_coordinates(results.right_hand_landmarks.landmark, 12)

                lfingers = fingersUp(results.right_hand_landmarks.landmark, 0)

                if identify_pose(lfingers, [2, 3]):
                    cv2.rectangle(image, (rx6, ry6 - 25), (rx7, ry7 + 25), drawColor, cv2.FILLED)
                    print("Left Selection Mode")
                    # Check for click
                    if ry6 < 125:
                        if 100 < rx6 < 200:
                            drawColor = (255, 0, 255)
                            header = overlayList[1]  # pencil tool
                        elif 250 < rx6 < 350:
                            drawColor = (255, 0, 255)
                            header = overlayList[2]  # Brush tool
                        elif 400 < rx6 < 510:
                            drawColor = (0, 0, 0)
                            header = overlayList[3]  # Shapes tool
                        elif 750 < rx6 < 850:
                            drawColor = (255, 0, 255)
                            header = overlayList[4]  # Undo tool
                        elif 900 < rx6 < 1000:
                            drawColor = (255, 0, 255)
                            header = overlayList[5]  # Redo tool
                        elif 1100 < rx6 < 1200:
                            drawColor = (255, 0, 255)
                            imgCanvas = np.zeros((hCam, wCam, 3), np.uint8)
                            header = overlayList[6]  # Delete tool
                            compare = False

            ## Right Hand
            if results.left_hand_landmarks is not None:

                rfingers = fingersUp(results.left_hand_landmarks.landmark, 1)

                if results.right_hand_landmarks is not None:
                    lfingers = fingersUp(results.right_hand_landmarks.landmark, 0)

                    if rfingers[1:] == [0, 1, 1, 1, 1] and lfingers[2:] == [0, 0, 0, 0]:
                        rx1, ry1 = get_coordinates(results.left_hand_landmarks.landmark, 0)
                        rx2, ry2 = get_coordinates(results.left_hand_landmarks.landmark, 8)
                        rx3, ry3 = get_coordinates(results.left_hand_landmarks.landmark, 12)

                        d2 = get_distance(rx2, ry2, rx3, ry3)
                        d3 = get_distance(rx1, ry1, rx3, ry3)

                        if d3 < 350:
                            image = create_menuWheel(image, results.left_hand_landmarks.landmark)

                            if abs(rx3 - (rx1 - 50)) < 20:
                                selectionCol = (255, 0, 0)
                                col1, col2, col3, col4 = selectionCol, tc2, tc3, tc4

                                image = add_desc(image, selectionCol,
                                                 heading='Normal Mode',
                                                 s1='A mode in which no special',
                                                 s2='gestures are used apart from',
                                                 s3='the menu gesture.')

                                if d2 < 30:
                                    print("Normal Mode")
                                    # countdown of 5
                                    return True

                            elif abs(rx3 - rx1) < 20:
                                selectionCol = (0, 0, 255)
                                col1, col2, col3, col4 = tc1, selectionCol, tc3, tc4

                                image = add_desc(image, selectionCol,
                                                 heading='Drawing Mode',
                                                 s1='A mode for drawing and',
                                                 s2='writing, creating shapes and',
                                                 s3='much more.')

                            elif abs(rx3 - (rx1 + 50)) < 20:
                                selectionCol = (0, 255, 255)
                                col1, col2, col3, col4 = tc1, tc2, selectionCol, tc4

                                image = add_desc(image, selectionCol,
                                                 heading='Canvas Mode',
                                                 s1='A mode for playing around',
                                                 s2='with elements, creating',
                                                 s3='new screens.')

                            elif abs(rx3 - (rx1 + 100)) < 20:
                                selectionCol = (0, 255, 0)
                                col1, col2, col3, col4 = tc1, tc2, tc3, selectionCol

                                image = add_desc(image, selectionCol,
                                                 heading='Demo Mode',
                                                 s1='A mode for working with',
                                                 s2='images and videos.')
                            else:
                                col1, col2, col3, col4 = tc1, tc2, tc3, tc4

                rx4, ry4 = get_coordinates(results.left_hand_landmarks.landmark, 8)
                rx5, ry5 = get_coordinates(results.left_hand_landmarks.landmark, 12)

                # Selection Using Right
                if identify_pose(rfingers, [2, 3]):
                    xp, yp = 0, 0
                    cv2.rectangle(image, (rx4, ry4 - 25), (rx5, ry5 + 25), drawColor, cv2.FILLED)
                    print("Right Selection Mode")
                    # Check for click
                    if ry4 < 125:
                        if 100 < rx4 < 200:
                            drawColor = (255, 0, 255)
                            header = overlayList[1]  # pencil tool
                        elif 250 < rx4 < 350:
                            drawColor = (255, 0, 255)
                            header = overlayList[2]  # Brush tool
                        elif 400 < rx4 < 510:
                            drawColor = (0, 0, 0)
                            header = overlayList[3]  # Shapes tool
                        # elif 550 < rx1 < 700:
                        #     drawColor = (255, 0, 255)
                        #     header = overlayList[4]  # Logo tool
                        elif 750 < rx4 < 850:
                            drawColor = (255, 0, 255)
                            header = overlayList[4]  # Undo tool
                        elif 900 < rx4 < 1000:
                            drawColor = (255, 0, 255)
                            header = overlayList[5]  # Redo tool
                        elif 1100 < rx4 < 1200:
                            drawColor = (255, 0, 255)
                            imgCanvas = np.zeros((hCam, wCam, 3), np.uint8)
                            header = overlayList[6]  # Delete tool


                brush_percent = 15
                # Pen Mode
                if (header == overlayList[1]).all() and identify_pose(rfingers, [2]) and checkHandexists(results,
                                                                                                         0) == False:
                    cv2.circle(image, (rx4, ry4), 15, drawColor, cv2.FILLED)
                    print("Drawing Mode")
                    if xp == 0 and yp == 0:
                        xp, yp = rx4, ry4
                    drawColor = (255, 0, 255)
                    brushThickness = int((brush_percent * 30) / 100)

                    cv2.line(image, (xp, yp), (rx4, ry4), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (rx4, ry4), drawColor, brushThickness)
                    xp, yp = rx4, ry4

                # Eraser Mode
                if identify_pose(rfingers, [2, 3, 4]) and checkHandexists(results, 0) == False:
                    drawColor = (0, 0, 0)

                    if xp == 0 and yp == 0:
                        xp, yp = rx4, ry4

                    brushThickness = int((brush_percent * 50) / 100) + 50

                    cv2.line(image, (xp, yp), (rx4, ry4), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (rx4, ry4), drawColor, brushThickness)

                    xp, yp = rx4, ry4

            # shape mode
            if (header == overlayList[3]).all():
                print("SHAPE MODEEEEE")
                drawColor = (225, 225, 225)
                cv2.rectangle(image, (600, 200), (1200, 500), drawColor, cv2.FILLED)
                cv2.rectangle(imgCanvas, (600, 200), (1200, 500), drawColor, cv2.FILLED)
                if identify_pose(rfingers, [1, 3]):
                    compare = True
                    # c = c + 1
                    # if c == 1:
                if compare:
                    pt1 = (700, 300)
                    pt2 = (700, 450)
                    pt3 = (850, 450)

                    cv2.circle(image, pt1, 2, (0, 0, 255), -1)
                    cv2.circle(image, pt2, 2, (0, 0, 255), -1)
                    cv2.circle(image, pt3, 2, (0, 0, 255), -1)

                    triangle_cnt = np.array([pt1, pt2, pt3])

                    cv2.drawContours(image, [triangle_cnt], 0, (0, 255, 0), -1)
                    cv2.drawContours(imgCanvas, [triangle_cnt], 0, (0, 255, 0), -1)
                    # header = overlayList[0]

                    # if c == 2:
                    cv2.putText(image, "hyp", (800, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                    cv2.putText(imgCanvas, "hyp", (800, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

                    cv2.putText(image, "base", (750, 480), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                    cv2.putText(imgCanvas, "base", (750, 480), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

                    cv2.putText(image, "alt", (650, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                    cv2.putText(imgCanvas, "alt", (650, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)


                    if identify_pose(rfingers, [1, 2]):
                        cv2.putText(image, "Pythagoras theorem", (850, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        cv2.putText(imgCanvas, "Pythagoras theorem", (850, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                        cv2.putText(image, "h = b + a", (920, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        cv2.putText(imgCanvas, "h = b + a", (920, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                        cv2.putText(image, "2", (930, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        cv2.putText(imgCanvas, "2", (930, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                        cv2.putText(image, "2", (1010, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        cv2.putText(imgCanvas, "2", (1010, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                        cv2.putText(image, "2", (1090, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        cv2.putText(imgCanvas, "2", (1090, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Combining with image
            # imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            # _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            # imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            # image = cv2.bitwise_and(image, imgInv)
            # image = cv2.bitwise_or(image, imgCanvas)

            # Setting the header image
            image[0:125, 0:1280] = header

            imgCanvas = cv2.addWeighted(imgCanvas, 0.5, image, 0.5, 0)  # blend with the canvas

            cv2.imshow("Canvas", image)
            # cv2.imshow("Canvas2", imgCanvas)

            cv2.waitKey(1)


def get_menu():
    folderPath = "menu"
    myList = os.listdir(folderPath)
    overlayList = [cv2.imread(f'{folderPath}/{imPath}') for imPath in myList]
    header = overlayList[0]
    return header, overlayList


def checkHandexists(results, hand):
    if hand == 0:  # left hand
        try:
            found = results.right_hand_landmarks.landmark
            return True
        except AttributeError:
            return False
    elif hand == 1:  # right hand
        try:
            found = results.left_hand_landmarks.landmark
            return True
        except AttributeError:
            return False


if __name__ == '__main__':
    detection_start()


# line
# circle
# square
# rectangle
# triangle
