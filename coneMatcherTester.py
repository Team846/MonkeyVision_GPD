import os
import cv2
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# USES TEST IMAGES TO DETECT CONES ONLY

def rotate_image(image, angle): #updated, now works with colored images!!! :)))
    if len(image.shape) == 2:  # for grascale images
        height, width = image.shape
        num_channels = 1
    elif len(image.shape) == 3:  # for colorful images
        height, width, num_channels = image.shape
    else:
        raise ValueError("Unsupported image format")

    bigger_size = int(np.ceil(np.sqrt(height**2 + width**2)))
    if num_channels == 1:
        bigger_background = np.zeros((bigger_size, bigger_size), dtype=image.dtype)
    else:
        bigger_background = np.zeros((bigger_size, bigger_size, num_channels), dtype=image.dtype)

    offset_x = (bigger_size - width) // 2
    offset_y = (bigger_size - height) // 2

    if num_channels == 1:
        bigger_background[offset_y:offset_y + height, offset_x:offset_x + width] = image
    else:
        bigger_background[offset_y:offset_y + height, offset_x:offset_x + width, :] = image

    image_center = (bigger_size // 2, bigger_size // 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    result = cv2.warpAffine(
        bigger_background,
        rot_mat,
        (bigger_size, bigger_size),
        flags=cv2.INTER_LINEAR
    )

    if num_channels == 1:
        return result, rot_mat, offset_x, offset_y, height, width
    else:
        return result, rot_mat,    offset_x, offset_y, height, width


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

upper_yellow = np.array([80, 255, 255], dtype = np.uint8)

lower_yellow = np.array([10, 100, 60], dtype = np.uint8)

def check_hsv(region):

    if len(region.shape) == 2:
        region = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
    

    # upper_yellow = np.array([255, 255, 255], dtype = np.uint8)

    # lower_yellow = np.array([0, 0, 0], dtype = np.uint8)


    hsvRegion = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsvRegion, lower_yellow, upper_yellow)


    yellow_pixels = cv2.countNonZero(mask)
    total_pixels = region.shape[0] * region.shape[1]
    percentage = (yellow_pixels/total_pixels)*100
    cv2.imwrite("r.jpg", mask)
    print("yellow pixels:")
    print(yellow_pixels)
    return percentage >= 30

def main(original_img_path):
    global lower_yellow, upper_yellow

    templates = load_templates("template", 3)

    original_img = cv2.imread(original_img_path)

    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (400, 300))

    highest = 0
    max_loc_ = (0, 0)
    i_highest = -1
    best_angle = 0

    t0 = time.time()

    for angle in range(0, 360, 45):
        rotated_img, _, offset_x, offset_y, height, width = rotate_image(gray_img, angle)
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

    best_rotated_img, best_rot_mat, offset_x, offset_y, height, width = rotate_image(gray_img, best_angle)
    best_rotated_img = cv2.cvtColor(best_rotated_img, cv2.COLOR_GRAY2BGR) 
    
    color_rotated_img, color_rot_mat, _, _, _, _ = rotate_image(original_img, best_angle)
    color_rotated_img = cv2.resize(color_rotated_img, (best_rotated_img.shape[1], best_rotated_img.shape[0]))


    cv2.imwrite("best_rotated.jpg", best_rotated_img)
    if highest > 0.55:
       
        templateX = templateUsed.shape[0]
        templateY = templateUsed.shape[1]
        
        # Note: subImage=Image[ miny:maxy, minx:maxx ]
        
        start_y = max(0, max_loc_[1] - templateY // 2)
        end_y = min(original_img.shape[0], max_loc_[1] + templateY // 2)

        start_x = max(0, max_loc_[0] - templateX // 2)
        end_x = min(original_img.shape[1], max_loc_[0] + templateX // 2)

        region = color_rotated_img[start_y:end_y, start_x:end_x]
        start_point = (start_x, start_y)
        end_point = (end_x, end_y)
        color = (255,0,0)
        thickness = 2
        cv2.circle(best_rotated_img, start_point, 5, (0, 0, 255), 2)

        mask = cv2.inRange(color_rotated_img, lower_yellow, upper_yellow)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask = cv2.resize(mask, (best_rotated_img.shape[1], best_rotated_img.shape[0]))

        print("MASK SHAPE")
        print(mask.shape)
        print("ROTATED IMG SHAPE")
        print(best_rotated_img.shape)
        print("COLOR ROTATED IMG SHAPE")
        print(color_rotated_img.shape)
        cv2.rectangle(mask, start_point, end_point, color, thickness)
        cv2.circle(mask, start_point, 5, (0, 0, 255), 2)

        print("START POINT" + (str)(start_point))

        if (region.shape[0] > 0 and region.shape[1] > 0 and check_hsv(region)):
            cv2.circle(best_rotated_img, max_loc_, 5, (255, 0, 255), 2)        
        cv2.rectangle(best_rotated_img, start_point, end_point, color, thickness)

        
        cv2.imwrite("mask.jpg", mask)
        inverse_rotation_matrix = cv2.invertAffineTransform(best_rot_mat)

        original_position_img = cv2.warpAffine(best_rotated_img, inverse_rotation_matrix, (original_img.shape[1], original_img.shape[0]))
        
        # remove the padding
        original_position_img = original_position_img[offset_y:offset_y + height, offset_x:offset_x + width]

        cv2.imshow("Original Position Image", original_position_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()    

        return original_position_img

if __name__ == "__main__":
    os.makedirs("testoutput", exist_ok=True)
    for imgnm in os.listdir("testimgs"):
        if imgnm.endswith(".jpg"):
            res = main(f"testimgs/{imgnm}")
            cv2.imwrite(f"testoutput/{imgnm}", res)

    # #LIVE CAMERA CODE: 
    # camera = cv2.VideoCapture(0)
    # while True:
    #     ret, frame = camera.read()
    #     frame = main(frame)
    #     if cv2.waitKey(1) == ord('q'):
    #         break

    cv2.destroyAllWindows()


