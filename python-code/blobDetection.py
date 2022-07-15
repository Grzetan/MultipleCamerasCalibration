import numpy as np
import cv2
import math
import os

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

def rotateImage(image, angleInDegrees):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return outImg, rot

def rotateKeypoints(keypoints, rotationMatrix):
    # There are minuses because y coordinate is inverted to match euclidean space
    for p in keypoints:
        point = (p.x, -p.y)
        rotatedPoint = rotationMatrix.dot(np.array(point + (1,)))
        p.x = rotatedPoint[0]
        p.y = -rotatedPoint[1]

    return keypoints

# Get average distance between neighbour blobs
def getBlobSpacing(grid):
    distances = []
    
    for row in grid:
        for i in range(len(row) - 1):
            for j in range(i+1, len(row)):
                distance = math.sqrt((row[i].x - row[j].x)**2 + (row[i].y - row[j].y)**2)
                difference = abs(i-j)
                distances.append(distance / difference)

    return sum(distances) / len(distances)

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

# Create grid using detected keypoints
def createGrid(startingBlob, keypoints):
    grid = []
    row, bellow = getRow(startingBlob, keypoints)
    grid.append(row)
    while(bellow is not None):
        row, bellow = getRow(bellow, keypoints)
        grid.append(row)

    return grid

# Get top left blob
def getTopLeftBlob(keypoints):
    topLeft = None
    min_ = np.inf

    for blob in keypoints:
        if blob.x + abs(blob.y) < min_:
            min_ = blob.x + abs(blob.y)
            topLeft = blob
    
    return topLeft

# Get camera rotation using detected grid
def getRotation(grid):
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

    return sum(angles) / len(angles)

# Calculate how much of calibration plate is covered in image
def getCoveredArea(grid, blobSpacing, shape, side):
    h, w = shape
    distances = []
    for row in grid:
        if side == 'LEFT':
            distances.append(w - row[0].x)
        else:
            distances.append(row[len(row) - 1].x)
    
    avg = sum(distances) / len(distances)
    blobsVisible = avg / blobSpacing
    return blobsVisible / (GRID_SIZE[0] - 1)

if __name__ == '__main__':
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
    
    # Input grid size of calibration plate
    GRID_SIZE = (19, 13)

    # Input distance between blobs in milimeters
    BLOB_SPACING = 15

    PATH = './sliced-imgs/'
    leftImgs = sorted([os.path.join(PATH, p) for p in os.listdir(PATH) if 'left' in p.lower()])
    rightImgs = sorted([os.path.join(PATH, p) for p in os.listdir(PATH) if 'right' in p.lower()])

    for left, right in zip(leftImgs, rightImgs):
        print("\n===== NEW SET OF IMAGES ======")
        imL = cv2.imread(left)
        imR = cv2.imread(right)

        # Detect blobs
        detectionsL = detector.detect(imL)
        detector.empty()
        detectionsR = detector.detect(imR)

        # Create list of detected keypoints
        keypointsL = []
        keypointsR = []

        for i, keypoint in enumerate(detectionsL):
            x,y = keypoint.pt
            keypointsL.append(Blob(x,-y,i))

        for i, keypoint in enumerate(detectionsR):
            x,y = keypoint.pt
            keypointsR.append(Blob(x,-y,i))

        # Get top left blob in each image
        startingBlobL = getTopLeftBlob(keypointsL)
        startingBlobR = getTopLeftBlob(keypointsR)

        # Create grids from detected keypoints
        gridL = createGrid(startingBlobL, keypointsL)
        gridR = createGrid(startingBlobR, keypointsR)

        # Calculate rotation of each image
        rotationL = getRotation(gridL)
        rotationR = getRotation(gridR)

        print(f'LEFT CAMERA ROTATION: {rotationL}')
        print(f'RIGHT CAMERA ROTATION: {rotationR}')

        # Rotate image and keypoints so grid is horizontal to X-axis
        imL, rotationMatrixL = rotateImage(imL, -rotationL)
        keypointsL = rotateKeypoints(keypointsL, rotationMatrixL)
        imR, rotationMatrixR = rotateImage(imR, -rotationR)
        keypointsR = rotateKeypoints(keypointsR, rotationMatrixR)

        # Get average distance between neighbour blobs in grid
        blobSpacingL = getBlobSpacing(gridL)
        blobSpacingR = getBlobSpacing(gridR)

        # Calculate how much of calibration plate is covered on each image
        coveredAreaL = getCoveredArea(gridL, blobSpacingL, imL.shape[:2], 'LEFT')
        coveredAreaR = getCoveredArea(gridR, blobSpacingR, imR.shape[:2], 'RIGHT')

        # Calculate translation between cameras
        translation = (coveredAreaL + coveredAreaR - 1) * -1
        translationInMM = translation * ((GRID_SIZE[0]-1) * BLOB_SPACING)

        print(f'TRANSLATION BETWEEN CAMERAS: {round(translationInMM,1)} mm')

        # Rotate detection the same way as keypoints
        for i, d in enumerate(detectionsL):
            d.pt = (keypointsL[i].x, -keypointsL[i].y)

        for i, d in enumerate(detectionsR):
            d.pt = (keypointsR[i].x, -keypointsR[i].y)

        im_with_keypointsL = cv2.drawKeypoints(imL, detectionsL, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        im_with_keypointsR = cv2.drawKeypoints(imR, detectionsR, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow("KeypointsL", im_with_keypointsL)
        cv2.imshow("KeypointsR", im_with_keypointsR)

        cv2.waitKey(0)
