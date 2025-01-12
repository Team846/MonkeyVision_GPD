import cv2
import numpy as np
import time

if __name__ == "__main__":

    img = cv2.imread("./testimgs/4.jpg")
    
    height, width  = img.shape[:2]
    # img = cv2.resize(img, (width//4, height//4))
    # print(width, height)

    cv2.imshow("img", img)

    startTime = time.perf_counter()


    img = cv2.GaussianBlur(img, (15, 15), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    output = img.copy()

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 150)
    print("NO CIRCLE YET")
    if circles is not None:
        print("CIRCLE POSSIBLY")
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            print("CIRCLE")
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        try:
            endTime = time.perf_counter()
            print(f"Processing time: {endTime - startTime} seconds")
            cv2.imshow("detections", output)
            cv2.imwrite("detections.jpg", output)
            cv2.waitKey(0)


        except Exception as e:
            print(f"Error: {e}")

        cv2.imwrite("output.jpg", output)
    # endTime = time.perf_counter()

    # print(f"Processing time: {endTime - startTime} seconds")
