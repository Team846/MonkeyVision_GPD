import cv2
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cam = cv2.VideoCapture(0)
for i in range(1000):
  print(i)
  ret, img = cam.read()
  cv2.imshow("image", img)
  cv2.waitKey(100)

