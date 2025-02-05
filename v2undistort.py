import numpy as np

import cv2


# camera 01: algae camera
# camera_matrices = np.array([[920.1598769,    0.,         685.36092797],
#  [  0.,         916.3391272,  368.03760649],
#  [  0.,           0.,           1.        ]])
# dist_coeffs = np.array([[ 0.01841649,  0.05107274,  0.00252806, -0.00028218, -0.12029582]])

#camera 02: yellow camera
camera_matrices = np.array([[913.7377659,    0.,         673.42504633],
 [  0.,         909.01346161, 464.51229778],
 [  0.,           0.,           1.        ]])
dist_coeffs = np.array([[ 0.03171734, -0.01147495, -0.00010437, -0.00082573, -0.059311  ]])

#camera 03: 


# avg_camera_matrix = np.mean(camera_matrices, axis=0)
# print("mtxs:", camera_matrices)
# avg_dist_coeffs = np.mean(dist_coeffs, axis=0)

# print("Averaged Camera Matrix:\n", avg_camera_matrix)
# print("Averaged Distortion Coefficients:\n", avg_dist_coeffs)

cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()

    if img is None:
        print("No image")

    h, w = img.shape[:2]

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrices, dist_coeffs, (w, h), 1, (w, h))

    undistorted_img = cv2.undistort(img, camera_matrices, dist_coeffs, None, new_camera_matrix)
    cv2.imwrite('v2undistorted.jpg', undistorted_img)
    cv2.imwrite('original.jpg', img)

