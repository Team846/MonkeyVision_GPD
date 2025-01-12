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

# runPipeline() is called every frame by Limelight's backend.
def runPipeline(frame):
    # Convert the input frame to the HSV color space
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    upper = np.array([90, 200, 255], dtype=np.uint8)
    lower = np.array([75, 75, 30], dtype=np.uint8)
    img_threshold = cv2.inRange(img_hsv, lower, upper)

    # Save the mask image for debugging purposes
    cv2.imwrite("mask.jpg", img_threshold)

    # Find contours in the binary image
    contours, _ = cv2.findContours(
        img_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) < 4:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contours[i])

        # filter by the radius size
        if radius < 50:
            continue

        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(frame, center, radius, (0, 255, 0), 2)  
        
        draw_point(frame, x, y)  # Draw the center point

        # pixels to angles
        tx, ty = px_to_deg(x, y)
        print(f"Ellipse {i}: center (tx, ty) = ({tx}, {ty}), radus = {radius}")

    return img_threshold, frame

if __name__ == "__main__":
    startTime = time.perf_counter()

    # Load the test image
    img = cv2.imread("./testimgs/4.jpg")

    # Apply Gaussian blur
    img = cv2.GaussianBlur(img, (101, 101), 0)

    # Run the pipeline
    threshold_img, processed_img = runPipeline(img)

    endTime = time.perf_counter()
    print(f"Processing time: {endTime - startTime} seconds")

    # Display the threshold and processed images
    cv2.imshow("Threshold", threshold_img)
    cv2.imshow("Processed", processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()