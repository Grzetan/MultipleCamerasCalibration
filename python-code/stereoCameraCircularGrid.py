import numpy as np
import os
import cv2

# https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html

def stereo_calibrate(gridSize, mtx1, dist1, mtx2, dist2, imgspath):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    left = sorted([p for p in os.listdir(imgspath) if 'left' in p])
    right = sorted([p for p in os.listdir(imgspath) if 'right' in p])

    c1_images = []
    c2_images = []
    for im1, im2 in zip(left, right):
        im1 = os.path.join(imgspath, im1)
        im2 = os.path.join(imgspath, im2)
        _im = cv2.imread(im1, 1)
        c1_images.append(_im)
 
        _im = cv2.imread(im2, 1)
        c2_images.append(_im)
  
    world_scaling = 1. #change this to the real world square size. Or not.
 
    objp = np.zeros((gridSize[0]*gridSize[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:gridSize[0],0:gridSize[1]].T.reshape(-1,2)
    objp = world_scaling* objp
 
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints_left = []
    imgpoints_right = []
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = []
 
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv2.findCirclesGrid(gray1, gridSize, None)
        c_ret2, corners2 = cv2.findCirclesGrid(gray2, gridSize, None)
 
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
 
            # cv2.drawChessboardCorners(frame1, gridSize, corners1, c_ret1)
            # cv2.imshow('img', frame1)
 
            # cv2.drawChessboardCorners(frame2, gridSize, corners2, c_ret2)
            # cv2.imshow('img2', frame2)
            # k = cv2.waitKey(500)
 
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
 
    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
 
    print("Calibrated both cameras with error: ", ret)
    return R, T

if __name__ == '__main__':
    gridSize = (19,13)

    stereo_calibrate(gridSize, None, None, None, None, 'stereo-imgs')
