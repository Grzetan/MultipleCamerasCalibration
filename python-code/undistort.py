import cv2
import numpy as np

FILE = './circles-imgs/1.jpg'
CAMERA_MATRIX_PATH = './cameraMatrix.npy'
DIST_PATH = './dist.npy'

mtx = np.load(CAMERA_MATRIX_PATH, allow_pickle=True)
dist = np.load(DIST_PATH, allow_pickle=True)

print(mtx, dist)

img = cv2.imread(FILE)
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
cv2.imwrite('calibresult.png', dst)