import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
last_time = 0
BG_COLOR = (255, 255, 255)
# cap = cv2.VideoCapture('other_assets/Pexels Videos 2675513.mp4')
cap = cv2.VideoCapture(0)
# For webcam input
with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
    bg_image = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            continue

        frame = cv2.flip(frame, 1)

        # segmentation code
        frame.flags.writeable = False
        results = selfie_segmentation.process(frame)
        frame.flags.writeable = True

        # apply joint bilateral filter
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1

        # Apply background magic
        # bg_image = cv2.imread('bg_asset/white.png')  # create a virtual bg
        # bg_image = cv2.GaussianBlur(frame, (55, 55), 0)  # blur current bg

        if bg_image is None:
            bg_image = np.zeros(frame.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR

        out_image = np.where(condition, frame, bg_image)  # return elements chosen from x or y depending on condition

        # Get frame rate
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("FPS: ", fps)

        # Get current time
        current_time = time.time()
        fps = 1 / (current_time - last_time)
        last_time = current_time
        cv2.putText(out_image, "FPS: {:.2f}".format(fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display results
        cv2.imshow('frame', out_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()