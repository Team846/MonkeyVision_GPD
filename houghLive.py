import cv2
import numpy as np
import time

def houghCircles(img):
    # img = cv2.GaussianBlur(img, (51, 51), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    cv2.imshow("gray", gray)
    output = img.copy()

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 50)
    # circles = cv2.HoughCircles(gray,  
    #                cv2.HOUGH_GRADIENT, 400, 2, param1 = 50, 
    #            param2 = 20, minRadius = 40)
    #print("NO CIRCLE YET")
    if circles is not None:
        #print("CIRCLE POSSIBLY")
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            #print("CIRCLE")
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        
        return output
    
    return img


if __name__ == "__main__":

    img = cv2.imread("./testimgs/4.jpg")
    #cam.set(cv2.CAP_PROP_FPS, 30)
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()

        if frame is None:
            print("NONE")
            continue
        # height, width = frame.shape[:2]
        # print(height, width)

        frame = houghCircles(frame)
        if frame is None:
            print("NONE2")

        cv2.imshow('result', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # cam.release()
        # cv2.destroyAllWindows()

    




    
    