import cv2
import re
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile, join

# read logo image
logo = cv2.imread("other_assets/opencv_logo.png")

# read the first frame of the video
cap = cv2.VideoCapture("other_assets/Pexels Videos 2675513.mp4")

# upper and lower range of HSV
lower = np.array([6, 10, 68])
upper = np.array([30, 36, 122])

# create kernel for image dilation
kernel = np.ones((3, 3), np.uint8)

# execute the couple of lines below everytime you run the following while loop
cnt = 0

# loop to load, pre-process and display the frames
while (True):
    ret, f = cap.read()

    # extract the area where we will place the logo
    # the dimensions of this area should match with those of the logo
    mini_frame = f[500:740, 875:1070, :]

    # create HSV image
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)

    # create binary mask
    mask = cv2.inRange(hsv, lower, upper)

    # perform dilation on the mask to reduce noise
    dil = cv2.dilate(mask, kernel, iterations=5)

    # extract the area where we will place the logo
    # create 3 channels
    mini_dil = np.zeros_like(mini_frame)
    mini_dil[:, :, 0] = dil[500:740, 875:1070]
    mini_dil[:, :, 1] = dil[500:740, 875:1070]
    mini_dil[:, :, 2] = dil[500:740, 875:1070]

    # copy image of logo
    logo_copy = logo.copy()

    # set pixel values to 1 where the pixel values of the mask is 0
    logo_copy[mini_dil == 0] = 1

    # set pixel values to 1 where the pixel values of the logo is 0
    logo_copy[logo == 0] = 1

    # set pixel values to 1 where the pixel values of the logo is not 1
    mini_frame[logo_copy != 1] = 1

    # merge images (array multiplication)
    mini_frame = mini_frame * logo_copy

    # insert logo in the frame
    f[500:740, 875:1070, :] = mini_frame

    cv2.rectangle(f, (600, 200), (1200, 500), (0, 0, 255), cv2.FILLED)

    # resize the frame (optional)
    f = cv2.resize(f, (480, 270), interpolation=cv2.INTER_AREA)


    # display frame
    cv2.imshow('frame', f)

    # save frame
    # cv2.imwrite(path+'frames/'+str(cnt)+'.png',f)
    cnt += 1

    if cv2.waitKey(20) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break