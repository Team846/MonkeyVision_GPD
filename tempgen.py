import cv2
import os

img = cv2.imread("./crop_img2.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (15, 15), 0)
#img2 = cv2.imread("IMG_1180.jpg")

height, width = img.shape[:2]

sizes = []
for x in [160, 240, 320, 360, 400, 440, 480, 520, 560, 600]: # set sizes here
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