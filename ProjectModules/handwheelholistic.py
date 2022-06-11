import math

import mediapipe as mp
import cv2
import time


pTime = 0
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

################################
wCam, hCam = 1280, 720
################################

cap.set(3, wCam)
cap.set(4, hCam)

tipIds = [4, 8, 12, 16, 20]
drawColor = (255, 0, 255)

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
    global pTime, drawColor
    col1, col2, col3, col4 = (243, 179, 147), (134, 144, 228), (117, 215, 247), (153, 199, 145)
    tc1, tc2, tc3, tc4 = col1, col2, col3, col4
    selectionCol = (0, 0, 255)
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

            # Start Choosing
            ## Right Hand
            if results.left_hand_landmarks is not None:

                rfingers = fingersUp(results.left_hand_landmarks.landmark, 1)

                if results.right_hand_landmarks is not None:
                    lfingers = fingersUp(results.right_hand_landmarks.landmark, 0)

                    if rfingers[1:] == [0, 1, 1, 1, 1] and lfingers[2:] == [0, 0, 0, 0]:
                        rx1, ry1 = int(results.left_hand_landmarks.landmark[0].x * 1280), int(
                            results.left_hand_landmarks.landmark[0].y * 700)
                        rx2, ry2 = int(results.left_hand_landmarks.landmark[8].x * 1280), int(
                            results.left_hand_landmarks.landmark[8].y * 700)
                        rx3, ry3 = int(results.left_hand_landmarks.landmark[12].x * 1280), int(
                            results.left_hand_landmarks.landmark[12].y * 700)

                        d1 = int(math.hypot(rx2 - rx1, ry2 - ry1))+50

                        cv2.circle(image, (rx1-50, (ry1-d1)+20), 12, col1, cv2.FILLED)
                        cv2.circle(image, (rx1, (ry1-d1)+10), 12,  col2, cv2.FILLED)
                        cv2.circle(image, (rx1+50, (ry1-d1)+10), 12,  col3, cv2.FILLED)
                        cv2.circle(image, (rx1 + 100, (ry1 - d1) + 30), 12,  col4, cv2.FILLED)

                        cv2.circle(image, (rx3, ry3-25), 17,  selectionCol)

                        # print(rx3, rx1-50)

                        if abs(rx3-(rx1-50)) < 20:
                            selectionCol = (255, 0, 0)
                            col1, col2, col3, col4 = selectionCol, tc2, tc3, tc4

                            cv2.putText(image, 'Normal Mode', (800, 50), cv2.FONT_HERSHEY_COMPLEX, 1, selectionCol, 2)
                            cv2.putText(image, 'A mode in which no special', (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        selectionCol, 1)
                            cv2.putText(image, 'gestures are used apart from', (800, 140), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        selectionCol, 1)
                            cv2.putText(image, 'the menu gesture.', (800, 180), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        selectionCol, 1)

                        elif abs(rx3-rx1) < 20:
                            selectionCol = (0, 0, 255)
                            col1, col2, col3, col4 = tc1, selectionCol, tc3, tc4

                            cv2.putText(image, 'Drawing Mode', (800, 50), cv2.FONT_HERSHEY_COMPLEX, 1, selectionCol, 2)
                            cv2.putText(image, 'A mode for drawing and', (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        selectionCol, 1)
                            cv2.putText(image, 'writing, creating shapes and', (800, 140), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        selectionCol, 1)
                            cv2.putText(image, 'much more.', (800, 180), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        selectionCol, 1)

                        elif abs(rx3-(rx1+50)) < 20:
                            selectionCol = (0, 255, 255)
                            col1, col2, col3, col4 = tc1, tc2, selectionCol, tc4

                            cv2.putText(image, 'Canvas Mode', (800, 50), cv2.FONT_HERSHEY_COMPLEX, 1, selectionCol, 2)
                            cv2.putText(image, 'A mode for playing around', (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        selectionCol, 1)
                            cv2.putText(image, 'with elements, creating', (800, 140), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        selectionCol, 1)
                            cv2.putText(image, 'new screens.', (800, 180), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        selectionCol, 1)
                        elif abs(rx3-(rx1+100)) < 20:
                            selectionCol = (0, 255, 0)
                            col1, col2, col3, col4 = tc1, tc2, tc3, selectionCol

                            cv2.putText(image, 'Demo Mode', (800, 50), cv2.FONT_HERSHEY_COMPLEX, 1, selectionCol, 2)
                            cv2.putText(image, 'A mode for working with', (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        selectionCol, 1)
                            cv2.putText(image, 'images and videos.', (800, 140), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        selectionCol, 1)
                        else:
                            col1, col2, col3, col4 = tc1, tc2, tc3, tc4

                        # print(rx1, (ry1-d1)+3)

                        ## Keep a point constant
                        # try:
                        #     cv2.circle(image, (crx1, cry1-25), 15, drawColor, cv2.FILLED)
                        # except UnboundLocalError:
                        #     crx1, cry1 = tuple((rx1, ry1))



            # FPS CODE
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(image, f'FPS: {int(fps)}', (40, 250), cv2.FONT_HERSHEY_COMPLEX,
                        1, (255, 0, 0), 3)

            cv2.imshow("Img", image)
            cv2.waitKey(1)


if __name__ == '__main__':
    detection_start()
