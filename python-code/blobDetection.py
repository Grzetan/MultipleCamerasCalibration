import numpy as np
import cv2

im = cv2.imread('./sliced-imgs/center.jpg')

params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 10
params.maxThreshold = 100

# Without this segmentation fault
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else : 
    detector = cv2.SimpleBlobDetector_create(params)

detector.empty()
keypoints = detector.detect(im)

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
