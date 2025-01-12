import cv2
import numpy as np
import time

def houghCircles(img):
    img = cv2.GaussianBlur(img, (15, 15), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    output = img.copy()

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 50)
    print("NO CIRCLE YET")
    if circles is not None:
        print("CIRCLE POSSIBLY")
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            print("CIRCLE")
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        try:
            cv2.imshow("detections", output)
            cv2.imwrite("detections.jpg", output)


        except Exception as e:
            print(f"Error: {e}")
        
        return output
    
    return img


if __name__ == "__main__":

    img = cv2.imread("./testimgs/4.jpg")
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FPS, 30)
    while True:
        ret, frame = cam.read()

        if frame is None:
            print("NONE")
            time.sleep(1)
            continue
        height, width = frame.shape[:2]
        print(height, width)

        #frame = houghCircles(frame)
        if frame is None:
            print("NONE2")

        cv2.imshow('result', frame)

        img = frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cam.release()
        cv2.destroyAllWindows()

    




    
    