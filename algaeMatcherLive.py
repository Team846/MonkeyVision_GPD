import os
import cv2
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# USES LIVE CAMERA FOOTAGE TO DETECT ALGAE ONLY

def load_templates(template_folder, num_templates):
    return [
        cv2.imread(f"{template_folder}/{i}.jpg", cv2.IMREAD_GRAYSCALE) for i in range(num_templates)
    ]

def process_template(index, img, templates):
    height, width = img.shape[:2]
    print("TEMPLATE")
    print(height, width)
    template = templates[index]
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    
    max_loc_centered = (
        max_loc[0] + template.shape[1] // 2,
        max_loc[1] + template.shape[0] // 2,
    )
    return max_val, max_loc_centered, index

#find algae teals
upper_teal = np.array([200, 255, 120], dtype = np.uint8)

lower_teal = np.array([90, 140, 320], dtype = np.uint8)

def check_hsv(region):

    if len(region.shape) == 2:
        region = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
    
    hsvRegion = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsvRegion, lower_teal, upper_teal)


    selected_pixels = cv2.countNonZero(mask)
    total_pixels = region.shape[0] * region.shape[1]
    percentage = (selected_pixels/total_pixels)*100
    
    return percentage >= 0

def main(original_img):
    #global lower_teal, upper_teal
    if original_img is None:
        print("NONE")
    templates = load_templates("template", 10)
    original_img = cv2.GaussianBlur(original_img, (15, 15), 0)

    
    if (original_img is None):
        return
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    width, height = original_img.shape[:2]
    gray_img = cv2.resize(gray_img, (height, width))
    gray_height, gray_width = gray_img.shape[:2]

    print(" HEIGHT: ")
    print(height)
    print(" WIDTH: ")
    print(width)
    
    print("GRAY HEIGHT: ")
    print(gray_height)
    print("GRAY WIDTH: ")
    print(gray_width)

    # highest = 0
    # max_loc_ = (0, 0)
    # i_highest = -1
    # best_angle = 0

    t0 = time.time()

    for angle in range(0, 360, 45):
        
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = executor.map(lambda tpl: process_template(tpl, original_img, templates), [i for i in range(len(templates))])
        
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

    if highest > 0.55:
       
        templateX = templateUsed.shape[0]
        templateY = templateUsed.shape[1]
        
        # Note: subImage=Image[ miny:maxy, minx:maxx ]
        
        start_y = max(0, max_loc_[1] - templateY // 2)
        end_y = min(original_img.shape[0], max_loc_[1] + templateY // 2)

        start_x = max(0, max_loc_[0] - templateX // 2)
        end_x = min(original_img.shape[1], max_loc_[0] + templateX // 2)

        region = original_img[start_y:end_y, start_x:end_x]
        start_point = (start_x, start_y)
        end_point = (end_x, end_y)
        color = (255,0,0)
        thickness = 2
        cv2.circle(original_img, start_point, 5, (0, 0, 255), 2)

        mask = cv2.inRange(original_img, lower_teal, upper_teal)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]))

        # print("MASK SHAPE")
        # print(mask.shape)
        # print("ROTATED IMG SHAPE")
        # print(best_rotated_img.shape)
        # print("COLOR ROTATED IMG SHAPE")
        # print(color_rotated_img.shape)
        cv2.rectangle(mask, start_point, end_point, color, thickness)
        cv2.circle(mask, start_point, 5, (0, 0, 255), 2)

        print("START POINT" + (str)(start_point))

        if (region.shape[0] > 0 and region.shape[1] > 0 and check_hsv(region)):
            cv2.circle(original_img, max_loc_, 5, (255, 0, 255), 2)        
        cv2.rectangle(original_img, start_point, end_point, color, thickness)

        
        cv2.imwrite("mask.jpg", mask)
        # inverse_rotation_matrix = cv2.invertAffineTransform(best_rot_mat)

        # original_position_img = cv2.warpAffine(best_rotated_img, inverse_rotation_matrix, (original_img.shape[1], original_img.shape[0]))
        
        # remove the padding
        # original_position_img = original_position_img[offset_y:offset_y + height, offset_x:offset_x + width]

        # cv2.imshow("Original Position Image", original_position_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()    

        return original_img
    else:
        return original_img
    
if __name__ == "__main__":
    os.makedirs("testoutput", exist_ok=True)
    # for imgnm in os.listdir("testimgs"):
    #     if imgnm.endswith(".jpg"):
    #         res = main(f"testimgs/{imgnm}")
    #         cv2.imwrite(f"testoutput/{imgnm}", res)

    camera = cv2.VideoCapture(0)
    Cwidth  = camera.get(3)  
    Cheight = camera.get(4)
    print(Cwidth, Cheight)

    while True:
        ret, frame = camera.read()

        frame = main(frame)
        cv2.imshow("result", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()