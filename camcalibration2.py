import numpy as np
import cv2 as cv
import glob
import os
import json

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

cam = cv.VideoCapture(0)

# images = glob.glob('*.jpg')
while True:
    ret, img = cam.read()

    if img is None:
        print("No image")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imwrite('img.jpg', img)
    # Find the chess board corners
    ret, corners =  cv.findChessboardCorners(gray, (6,9), None)

    # If found, add object points, image points (after refining them)
    file_num = 9
    directory = 'calibration_results'
    file_to_check = f'calibration_results/img_without_calib{file_num}.jpg'
    if ret == True:
        objpoints.append(objp)
        img_without_lines = img.copy()

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        
        #print(imgpoints)

        cv.drawChessboardCorners(img, (6,9), corners2, ret)
        cv.imwrite('./calibration/img.jpg', img)
        cv.imwrite(f'calibration_results/img_without_lines{file_num}.jpg', img_without_lines)

        # break

    cv.imwrite('./calibration/img.jpg', img)
    cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
img = cv.imread('calibration_results/img_without_lines3.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)