import numpy as np
import cv2
import math

class Blob(object):
    def __init__(self, x,y,idx):
        self.x = x
        self.y = y
        self.idx = idx

    def __repr__(self):
        return f"Blob(x: {self.x}, y: {self.y}, idx: {self.idx})"

class Line(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def calculateUsingOnePoint(self, blob, a):
        self.a = a
        self.b = -self.a * blob.x + blob.y
        
    def getY(self, x):
        return self.a * x + self.b

    def getTheta(self):
        return math.degrees(math.atan(self.a))

    def calculateUsingTwoPoints(self, blob1, blob2):
        self.a = (blob1.y - blob2.y) / ((blob1.x - blob2.x) + 1e-8) # For numerical stability
        self.b = -self.a * blob1.x + blob1.y

    def __repr__(self):
        return f"Line(y = {self.a}x + {self.b})"

def findClosestBlobs(currBlob, keypoints):
    # Find index of closest blob
    min_dist = 10000000
    closestBlob = None
    for blob in keypoints:
        if blob.idx == currBlob.idx: continue
        
        dist = math.sqrt((blob.x - currBlob.x)**2 + (blob.y - currBlob.y)**2)
        if dist < min_dist:
            min_dist = dist
            closestBlob = blob

    # Get blobs that are closest (blob to the right, left, top and bottom side)
    closest = []
    for blob in keypoints:
        if blob.idx == currBlob.idx: continue
        dist = math.sqrt((blob.x - currBlob.x)**2 + (blob.y - currBlob.y)**2)
        if abs(dist - min_dist) < 10:
            closest.append(blob)

    # Calculate relative position of closest blobs using two lines
    
    # Growing line crossing currBlob
    growing = Line(1, 0)
    growing.calculateUsingOnePoint(currBlob, 1)

    # Decreasing line crossing currBlob
    decreasing = Line(-1, 0)
    decreasing.calculateUsingOnePoint(currBlob, -1)

    left = None
    right = None
    top = None
    bottom = None

    for blob in closest:
        # Calculate if blob is grater than lines
        greaterThanGrowing = blob.y > growing.getY(blob.x)
        greaterThanDecreasing = blob.y > decreasing.getY(blob.x)

        # Decide using calcuated flags relative position of each closest blob
        if greaterThanDecreasing and not greaterThanGrowing:
            right = blob
        elif greaterThanGrowing and greaterThanDecreasing:
            top = blob
        elif not greaterThanGrowing and not greaterThanDecreasing:
            bottom = blob
        elif greaterThanGrowing and not greaterThanDecreasing:
            left = blob

    return [top, right, bottom, left]

# Get all blobs to the right of passed node and return this arrays along side with blob bellow
def getRow(currBlob, keypoints):
    closest = findClosestBlobs(currBlob, keypoints)
    row = [currBlob]

    # As long as there is a blob to the right
    while closest[1] is not None:
        row.append(closest[1])
        closest = findClosestBlobs(closest[1], keypoints)

    return row, findClosestBlobs(currBlob, keypoints)[2]

im = cv2.imread('./sliced-imgs/rotatedLeft.jpg')

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
detections = detector.detect(im)
keypoints = []

for i, keypoint in enumerate(detections):
    x,y = keypoint.pt
    keypoints.append(Blob(x,-y,i))

# Find index of top left blob on calibration plate
topLeft = None
min_ = 1000000

for blob in keypoints:
    if blob.x + abs(blob.y) < min_:
        min_ = blob.x + abs(blob.y)
        topLeft = blob

grid = []
row, bellow = getRow(topLeft, keypoints)
grid.append(row)
while(bellow is not None):
    row, bellow = getRow(bellow, keypoints)
    grid.append(row)

# Calculate rotation of each row and use average as camera's rotation
angles = []
for row in grid:
    # Get line between every possible combination of points in row and take average of parameters as row's line
    aSet = []
    bSet = []
    line = Line(0,0) # Parameters change
    for i in range(len(row) - 1):
        for j in range(i+1, len(row)):
            line.calculateUsingTwoPoints(row[i], row[j])
            aSet.append(line.a)
            bSet.append(line.b)
    
    line.a = sum(aSet) / len(aSet)
    line.b = sum(bSet) / len(bSet)
    angles.append(line.getTheta())

print(f'CAMERA ROTATION: {sum(angles) / len(angles)}')

det = []
for i, detection in enumerate(detections):
    if i in [p.idx for p in grid[4]]:
        det.append(detection)

# print(findClosestBlobs(keypoints[0], keypoints))

im_with_keypoints = cv2.drawKeypoints(im, detections, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
