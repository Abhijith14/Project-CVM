# import dependencies
import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
import cv2
from matplotlib import pyplot as plt
import numpy as np

# Detections
bodypix_model = load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))

# opencv video capture
cap = cv2.VideoCapture(0)

################################
wCam, hCam = 1280, 960
################################

cap.set(3, wCam)
cap.set(4, hCam)

img = cv2.imread('bg_asset/white.png')
img = img[:720, :1280, :]

print(img.shape)



while cap.isOpened():
    ret, frame = cap.read()

    # bodypix detections
    result = bodypix_model.predict_single(frame)
    mask = result.get_mask(threshold=0.5).numpy().astype(np.uint8)
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)


    # Add virtual background

    # img = cv2.resize(img, (wCam, hCam))
    print(frame.shape)
    neg = np.add(mask, -1)
    inverse = np.where(neg == -1, 1, neg).astype(np.uint8)
    masked_background = cv2.bitwise_and(img, img, mask=inverse)
    final = cv2.add(masked_image, masked_background)


    # frame = cv2.flip(frame, 1)
    # show the frame
    cv2.imshow('frame', final)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()