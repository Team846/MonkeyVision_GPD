import os
import cv2
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

##USES LIVE CAMERA FOOTAGE TO DETECT 

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def load_templates(template_folder, num_templates):
    return [
        cv2.imread(f"{template_folder}/{i}.jpg", cv2.IMREAD_GRAYSCALE) for i in range(num_templates)
    ]

def process_template(template, img, threshold=0.55):
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val > threshold:
        max_loc_centered = (
            max_loc[0] + template.shape[1] // 2,
            max_loc[1] + template.shape[0] // 2,
        )
        return max_val, max_loc_centered
    return None, None

def main(original_img):
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
            results = executor.map(lambda tpl: process_template(tpl, rotated_img), templates)

        for i, (max_val, max_loc_centered) in enumerate(results):
            if max_val and max_val > highest:
                highest = max_val
                max_loc_ = max_loc_centered
                i_highest = i
                best_angle = angle

    t1 = time.time()

    print(f"Execution time: {t1 - t0:.2f} seconds")
    print("Highest match value:", highest)
    print("Index of best match:", i_highest)
    print("Best angle:", best_angle)

    best_rotated_img = rotate_image(original_img, best_angle)
    best_rotated_img = cv2.cvtColor(best_rotated_img, cv2.COLOR_GRAY2BGR)
    if highest > 0.55:
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


