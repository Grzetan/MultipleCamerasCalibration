import numpy as np
import cv2
import math

im = cv2.imread('./sliced-imgs/rotatedRight.jpg')

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

# Find index of top left blob on calibration plate
idx = 0
min_ = 1000000

for i, keypoint in enumerate(keypoints):
    x,y = keypoint.pt
    if x+y < min_:
        min_ = x+y
        idx = i

# Find index of blob to the right of top left blob
min_dist = 10000000
x, y = keypoints[idx].pt
idx2 = 0
for i, keypoint in enumerate(keypoints):
    if i == idx: continue
    x1, y1 = keypoints[i].pt
    dist = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    if dist < min_dist:
        min_dist = dist
        idx2 = i

im_with_keypoints = cv2.drawKeypoints(im, [keypoints[idx2]], np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
