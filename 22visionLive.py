import cv2
import numpy as np
import time

# convert from pixels to angles
# screen width (px) = 320
# screen height (px) = 240
# screen FOV x (deg) = 59.6
# screen FOV y (deg) = 49.7
def px_to_deg(cx, cy):
    tx = ((cx - 160.0) / 320.0) * 59.6
    ty = ((cy - 120.0) / 240.0) * 49.7
    return tx, -ty

def draw_point(image, x, y):
    cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), cv2.FILLED)

def runPipeline(frame):
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    upper = np.array([85, 200, 255], dtype=np.uint8)
    lower = np.array([75, 80, 20], dtype=np.uint8)
    
    img_threshold = cv2.inRange(img_hsv, lower, upper)

    cv2.imwrite("mask.jpg", img_threshold)

    contours, _ = cv2.findContours(
        img_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) < 4:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contours[i])

        # filtesr by the radius size
        if radius < 50:
            continue

        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(frame, center, radius, (0, 255, 0), 2)  
        
        draw_point(frame, x, y) 

        # pixels to angles
        tx, ty = px_to_deg(x, y)
        print(f"Ellipse {i}: center (tx, ty) = ({tx}, {ty}), radus = {radius}")

    return img_threshold, frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        
        ret, img = cap.read()
        if not ret:
            break

        startTime = time.perf_counter()

        # add gausian blur
        img = cv2.blur(img, (51, 51), 0)
        
        # run the pipeline
        _, processed_img, = runPipeline(img)
        
        endTime = time.perf_counter()
        print(f"Processing time: {endTime - startTime} seconds")

        # display the output
        cv2.imshow("output", processed_img)
        
        # break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()