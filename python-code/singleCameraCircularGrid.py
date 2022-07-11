import numpy as np
import cv2 
import os

# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

print('OpenCV Version: ', cv2.__version__)

# gridSize = (10,7)
gridSize = (13, 19)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((gridSize[0] * gridSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:gridSize[0], 0:gridSize[1]].T.reshape(-1, 2)

objPoints = []
imgPoints = []

images = os.listdir('./circles-imgs')

for imgName in images:
    print(imgName)
    img = cv2.imread(os.path.join('./circles-imgs', imgName))
    h, w = img.shape[0], img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findCirclesGrid(gray, gridSize, None)
    if ret == True:
        objPoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgPoints.append(corners)

        cv2.drawChessboardCorners(img, gridSize, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(0)

# cv2.destroyAllWindows()

# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)

# img = cv2.imread('./org/left07.jpg')
# h, w = img.shape[:2]
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# # crop the image
# # x, y, w, h = roi
# # dst = dst[y:y+h, x:x+w]
# cv2.imwrite('calibresult.png', dst)

# mean_error = 0
# for i in range(len(objPoints)):
#     imgpoints2, _ = cv2.projectPoints(objPoints[i], rvecs[i], tvecs[i], mtx, dist)
#     error = cv2.norm(imgPoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
#     mean_error += error
# print( "total error: {}".format(mean_error/len(objPoints)) )