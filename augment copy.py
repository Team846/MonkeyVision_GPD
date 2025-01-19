import cv2
import os
import numpy as np

def read_images_from_folder(folder_path):
   images = []
   for file in os.listdir(folder_path):
       if file.endswith("jpg"):
           img_path = os.path.join(folder_path, file)
      #     img = cv2.imread(img_path)
           images.append(img_path)
   return images

ALGAE_IMAGES = "/Users/rayming/MonkeyVision_GPD/ALGAE_IMAGES"
images = read_images_from_folder(ALGAE_IMAGES)
# ALGAE_IMAGES_2 = "/Users/rayming/MonkeyVision_GPD/ALGAE_IMAGES_TEST"
# images2 = read_images_from_folder(ALGAE_IMAGES_2)
# ALGAE_IMAGES_3 = "/Users/rayming/MonkeyVision_GPD/ALGAE_IMAGES_TEST"
# images3 = read_images_from_folder(ALGAE_IMAGES_3)
# ALGAE_IMAGES_4 = "/Users/rayming/MonkeyVision_GPD/ALGAE_IMAGES_TEST"
# images4 = read_images_from_folder(ALGAE_IMAGES_4) 

ALGAE_IMAGES_AUGMENTED = "/Users/rayming/MonkeyVision_GPD/ALGAE_IMAGES_AUGMENTED"
os.makedirs(ALGAE_IMAGES_AUGMENTED, exist_ok=True)

def enumerate_images(images):
    blurs = [0, 20, 50]
    blur_name = ["noblur_", "blur1_", "blur2_"]
    lighting_factors = [-75, -50, -25, 0, 25, 50, 75]
    lighting_name = ["darkness3_", "darkness2_", "darkness1_", "normallighting_", "brightness1_", "brightness2_", "brightness3_"]
    color_shift_factors = [0, -10, 10]
    color_shift_name = ["nocolorshift_", "colorshift1_", "colorshift2_"]
    for i, img in enumerate(images):
        for j in range(3):
            for k in range(7):
                for l in range(3):
                    
                    if (l != 0):
                        blurred_img = cv2.blur(img, (blurs[l], blurs[l]))
                    else:
                        blurred_img = img
                    tag1 = blur_name[l]
                    brightened_img = cv2.addWeighted(blurred_img, 1, blurred_img, 0, lighting_factors[k])
                    tag2 = lighting_name[k]
                    hsv_image = cv2.cvtColor(brightened_img, cv2.COLOR_BGR2HSV)
                    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + color_shift_factors[j]) % 180
                    random_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
                    tag3 = color_shift_name[j]

                    new_image_path = os.path.join(ALGAE_IMAGES_AUGMENTED, f"algae_image_1_{i}{j}{k}{l}.jpg")
                    cv2.imwrite(new_image_path, random_image)


ALGAE_IMAGES_AUGMENTED_RESIZED = "/Users/rayming/MonkeyVision_GPD/ALGAE_IMAGES_AUGMENTED_RESIZED"


ALGAE_IMAGES_AUGMENTED_2 = "/Users/rayming/MonkeyVision_GPD/ALGAE_IMAGES_AUGMENTED_2"
augment_images = read_images_from_folder(ALGAE_IMAGES_AUGMENTED_2)

for i, img_path in enumerate(augment_images):
   img = cv2.imread(img_path)
   height, width = img.shape[:2]
   new_width = int (width/5)
   new_height = int (height/5)


   resized_image = cv2.resize(img, (new_width, new_height))


   new_image_path = os.path.join(ALGAE_IMAGES_AUGMENTED_RESIZED, f"algae_image_2_{i}.jpg")
   cv2.imwrite(new_image_path, resized_image)

""" # 2 blurred 1 normal
for i, img in enumerate(images):

    blurred_img = cv2.blur(img, (20, 20))
    blurred_img2 = cv2.blur(img, (50, 50))
    new_image_path = os.path.join(ALGAE_IMAGES_AUGMENTED, f"image_normal_{i}.jpg")
    new_image_path2 = os.path.join(ALGAE_IMAGES_AUGMENTED, f"image_blur_{i}.jpg")
    new_image_path3 = os.path.join(ALGAE_IMAGES_AUGMENTED, f"image_blur2_{i}.jpg")
    cv2.imwrite(new_image_path, img)
    cv2.imwrite(new_image_path2, blurred_img) 
    cv2.imwrite(new_image_path3, blurred_img2) 

# 3 brightened
for i, img in enumerate(images):
    brightened_image = cv2.addWeighted(img, 1, img, 0, 25)
    brightened2_image = cv2.addWeighted(img, 1, img, 0, 50)
    brightened3_image = cv2.addWeighted(img, 1, img, 0, 75)
    new_image_path = os.path.join(ALGAE_IMAGES_AUGMENTED, f"image_brightened_{i}.jpg")
    new_image_path2 = os.path.join(ALGAE_IMAGES_AUGMENTED, f"image_brightened2_{i}.jpg")
    new_image_path3 = os.path.join(ALGAE_IMAGES_AUGMENTED, f"image_brightened3_{i}.jpg")
    cv2.imwrite(new_image_path, brightened_image)
    cv2.imwrite(new_image_path2, brightened2_image)
    cv2.imwrite(new_image_path3, brightened3_image) 

 # 3 darkened
for i, img in enumerate(images):
    darkened_image = cv2.addWeighted(img, 1, img, 0, -25)
    darkened2_image = cv2.addWeighted(img, 1, img, 0, -50)
    darkened3_image = cv2.addWeighted(img, 1, img, 0, -75)
    new_image_path = os.path.join(ALGAE_IMAGES_AUGMENTED, f"image_darkened_{i}.jpg")
    new_image_path2 = os.path.join(ALGAE_IMAGES_AUGMENTED, f"image_darkened2_{i}.jpg")
    new_image_path3 = os.path.join(ALGAE_IMAGES_AUGMENTED, f"image_darkened3_{i}.jpg")
    cv2.imwrite(new_image_path, darkened_image)
    cv2.imwrite(new_image_path2, darkened2_image)
    cv2.imwrite(new_image_path3, darkened3_image) 
    

# 2 slight color shifts
for i, img in enumerate(images):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    shift_value = 10
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + shift_value) % 180
    shifted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    new_image_path = os.path.join(ALGAE_IMAGES_AUGMENTED, f"image_color_shifted_{i}.jpg")
    cv2.imwrite(new_image_path, shifted_image)

    hsv_image_2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    shift_value = -10
    hsv_image_2[:, :, 0] = (hsv_image_2[:, :, 0] + shift_value) % 180
    shifted_image2 = cv2.cvtColor(hsv_image_2, cv2.COLOR_HSV2BGR)
    new_image_path2 = os.path.join(ALGAE_IMAGES_AUGMENTED, f"image_color_shifted2_{i}.jpg")
    cv2.imwrite(new_image_path2, shifted_image2) """

