import os
import cv2
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# USES LIVE CAMERA FOOTAGE TO DETECT CONES ONLY

def rotate_image(image, angle):
    height = image.shape[0]
    width = image.shape[1]

    biggerSize = (int) (np.sqrt(height**2 + width**2))
    biggerBackground = np.zeros((biggerSize, biggerSize), dtype=image.dtype)
    offset_x = (biggerSize - width)//2
    offset_y = (biggerSize - height)//2

    biggerBackground[offset_y:offset_y+height, offset_x:offset_x+width] = image

    image_center = (biggerSize//2, biggerSize//2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    result = cv2.warpAffine(biggerBackground, rot_mat, (biggerSize, biggerSize), flags=cv2.INTER_LINEAR)
    return result

def load_templates(template_folder, num_templates):
    return [
        cv2.imread(f"{template_folder}/{i}.jpg", cv2.IMREAD_GRAYSCALE) for i in range(num_templates)
    ]

def process_template(index, img, templates):
    template = templates[index]
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    
    max_loc_centered = (
        max_loc[0] + template.shape[1] // 2,
        max_loc[1] + template.shape[0] // 2,
    )
    return max_val, max_loc_centered, index
    

def check_hsv(region):

    if len(region.shape) == 2:
        region = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
    
    upper_yellow = np.array([40, 255, 255], dtype = np.uint8)

    lower_yellow = np.array([0, 40, 60], dtype = np.uint8)


    hsvRegion = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsvRegion, lower_yellow, upper_yellow)


    yellow_pixels = cv2.countNonZero(mask)
    total_pixels = region.shape[0] * region.shape[1]
    percentage = (yellow_pixels/total_pixels)*100
    cv2.imshow("r", mask)
    cv2.imshow("region", region)

    return percentage >= 5

def main(original_img):
    global lower_yellow, upper_yellow

    templates = load_templates("template", 3)

    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    original_img = cv2.resize(original_img, (400, 300))

    highest = 0
    max_loc_ = (0, 0)
    i_highest = -1
    best_angle = 0

    t0 = time.time()

    for angle in range(0, 360, 45):
        rotated_img = rotate_image(original_img, angle)
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = executor.map(lambda tpl: process_template(tpl, rotated_img, templates), [i for i in range(len(templates))])
    

        for _,(max_val, max_loc_centered, i) in enumerate(results):
            if max_val and max_val > highest:
                highest = max_val
                max_loc_ = max_loc_centered
                i_highest = i
                best_angle = angle
                templateUsed = templates[i_highest]
                

    t1 = time.time()

    print(f"Execution time: {t1 - t0:.2f} seconds")
    print("Highest match value:", highest)
    print("Index of best match:", i_highest)
    print("Best angle:", best_angle)

    best_rotated_img = rotate_image(original_img, best_angle)
    best_rotated_img = cv2.cvtColor(best_rotated_img, cv2.COLOR_GRAY2BGR)
    if highest > 0.55:
       
        templateX = templateUsed.shape[0]
        print(templateX) #remove later
        templateY = templateUsed.shape[1]
        print(templateY) #remove later
        
        # Note: subImage=Image[ miny:maxy, minx:maxx ]
        print(max_loc_[0])
        print(max_loc_[1])
        print(original_img.shape[0])
        print(original_img.shape[1])
        cv2.imshow("o", original_img)
        region = original_img[max_loc_[0] - templateY//2: max_loc_[0] + templateY//2, 
                              max_loc_[1] - templateX//2: max_loc_[1] + templateX//2]

        if (region.shape[0] > 0 and region.shape[1] > 0 and check_hsv(region)):
            cv2.circle(best_rotated_img, max_loc_, 5, (0, 0, 255), 2)

    cv2.imshow("Result", best_rotated_img)
    

    return best_rotated_img

if __name__ == "__main__":
    os.makedirs("testoutput", exist_ok=True)
    # for imgnm in os.listdir("testimgs"):
    #     if imgnm.endswith(".jpg"):
    #         res = main(f"testimgs/{imgnm}")
    #         cv2.imwrite(f"testoutput/{imgnm}", res)

    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()

        frame = main(frame)


        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()