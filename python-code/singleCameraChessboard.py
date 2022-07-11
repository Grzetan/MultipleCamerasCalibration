import numpy as np
import cv2 
import os

# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

print('OpenCV Version: ', cv2.__version__)

chessboardSize = (7,6)
frameSize = (1532, 2048)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

objPoints = []
imgPoints = []

images = os.listdir('./xd')

for img in images:
    print(img)
    img = cv2.imread(os.path.join('./xd', img))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize, flags=cv2.CALIB_CB_ADAPTIVE_THREASH)
    print(ret)
    if ret == True:
        objPoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgPoints.append(corners)

        cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(1000)

cv2.destroyAllWindows()