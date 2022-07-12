import numpy as np
import cv2 
import os

# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

def calibrate(chessboardSize, imgspath, savePath=''):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

    objPoints = []
    imgPoints = []

    images = os.listdir(imgspath)

    for imgName in images:
        img = cv2.imread(os.path.join(imgspath, imgName))
        h, w = img.shape[0], img.shape[1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)
        if ret == True:
            objPoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgPoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)

    print('Calibrated camera with error (Should be less than 0.2): ', ret)

    if savePath != '':
        np.savez(savePath, mtx=mtx, dist=dist)

    return (ret, mtx, dist, rvecs, tvecs)


if __name__ == '__main__':
    print('OpenCV Version: ', cv2.__version__)

    chessboardSize = (7,6)

    ret, mtx, dist, rvecs, tvecs = calibrate(chessboardSize, './org')

    img = cv2.imread('./org/left07.jpg')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult.png', dst)