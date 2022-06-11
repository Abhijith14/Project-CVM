import time
import cv2
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


pTime = 0
# cap = cv2.VideoCapture('http://192.168.29.201:4747/video')
# cap = cv2.VideoCapture('other_assets/Pexels Videos 2675513.mp4')
cap = cv2.VideoCapture(0)
currmin, currmax  = 60000, 0

################################
wCam, hCam = 1280, 960
################################

cap.set(3, wCam)
cap.set(4, hCam)


def set_color(color, t, r): # color - BGR, thickness, circle_radius
    return mp_drawing.DrawingSpec(color=color, thickness=t, circle_radius=r)


def add_fps(image):
    global pTime, currmin, currmax
    # FPS CODE
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # find the max fps

    if fps > currmax:
        currmax = fps

    # find the min fps
    if fps < currmin:
        currmin = fps

    print("Max = ", currmax, " --- Min = ", currmin, " --- FPS = ", fps)

    # cv2.putText(image, f'FPS: {int(fps)}', (40, 250), cv2.FONT_HERSHEY_COMPLEX,
    #             1, (255, 0, 0), 3)
    # return image

with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    while True:
        ret, image = cap.read()
        add_fps(image)

        # image.flags.writeable = False
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        # image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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


        cv2.imshow("Canvas", image)
        cv2.waitKey(1)
    cap.release()