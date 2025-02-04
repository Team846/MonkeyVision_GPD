import numpy as np
import json 
import cv2

with open('matrix.json', 'r') as f:
    data = json.load(f)
    
camera_matrices = []
dist_coeffs = []

for calib in data:
    cam_mat = np.array(calib['mtx'])
    dist_coef = np.array(calib['dist'])
    
    camera_matrices.append(cam_mat)
    dist_coeffs.append(dist_coef)
    
avg_camera_matrix = np.mean(camera_matrices, axis=0)
avg_dist_coeffs = np.mean(dist_coeffs, axis=0)

print("Averaged Camera Matrix:\n", avg_camera_matrix)
print("Averaged Distortion Coefficients:\n", avg_dist_coeffs)

img = cv2.imread('calibration_results/img_without_lines1.jpg')
if img is None:
    raise IOError("Image not found. Please check the file path.")

h, w = img.shape[:2]

new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(avg_camera_matrix, avg_dist_coeffs, (w, h), 1, (w, h))

undistorted_img = cv2.undistort(img, avg_camera_matrix, avg_dist_coeffs, None, new_camera_matrix)
cv2.imwrite('undistorted.jpg', undistorted_img)
