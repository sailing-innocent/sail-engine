import cv2 as cv 
import numpy as np 

cap = cv.VideoCapture(0)

while(1):
    _, frame = cap.read()

    print(frame)
    # convert bgr to hsv
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    mask = cv.inRange(hsv, lower_blue, upper_blue)

    res = cv.bitwise_and(frame, frame, mask = mask)

    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', res)
    k = cv.waitKey(5) & 0xFF

    if k == 27:
        break
    
cv.destroyAllWindows()