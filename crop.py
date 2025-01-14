import cv2

img = cv2.imread("testimgs/4.jpg")

rect = cv2.selectROI(img)

crop_img = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

cv2.imshow("crop_img", crop_img)
cv2.imwrite("crop_img2.jpg", crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
