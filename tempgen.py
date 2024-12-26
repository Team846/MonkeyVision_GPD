import cv2
import os

img = cv2.imread("crop_img.jpg")
img2 = cv2.imread("crop_img2.jpg")

height, width = img.shape[:2]

sizes = []
for x in [48, 64, 80]: # set sizes here
    y = int(x/2 + ((x / width) * height)/2)
    sizes.append((x, y))

if not os.path.exists("template"):
    os.makedirs("template")
else:
    for f in os.listdir("template"):
        os.remove(os.path.join("template", f))


cntr = 0
for size in sizes:
    resized = cv2.resize(img, size)

    filename = f"template/{cntr}.jpg"
    cv2.imwrite(filename, resized)
    cntr += 1
# for size in sizes:
#     resized = cv2.resize(img2, size)

#     filename = f"template/{cntr}.jpg"
#     cv2.imwrite(filename, resized)
#     cntr += 1