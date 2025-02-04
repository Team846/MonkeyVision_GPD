import numpy as np

import cv2
   
camera_matrices = np.array([[922.21332721,   0.0,         682.2337992 ],
 [  0.0,         920.19486032, 366.35006633],
 [  0.0,           0.0,           1.0        ]])
dist_coeffs = np.array([[ 0.03312957, -0.00165267,  0.00165053, -0.00130589, -0.11381045]])

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

