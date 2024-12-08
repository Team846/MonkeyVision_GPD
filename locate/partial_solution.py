from util.config import ConfigCategory
from util.logger import Logger
from cv2.typing import MatLike
from typing import List, Tuple

import math

import json
import os

TAG_FILE_PATH = "tags.json"

logger = Logger("PartialSolution")

loaded_tags = {}
if not os.path.exists(TAG_FILE_PATH):
    logger.Warn("Couldn't find existing AprilTag data file")
with open(TAG_FILE_PATH, 'r') as tag_file:
    loaded_tags = json.load(tag_file)
    logger.Log(f"Loaded tags: {loaded_tags}")

pref_category = ConfigCategory("PartialSolution")

CAM_FOV_X = pref_category.getFloatConfig("CAM_FOV_X_deg", 70.0)
CAM_FOV_Y = pref_category.getFloatConfig("CAM_FOV_Y_deg", 47.27)
CAM_ANGLE_H = pref_category.getFloatConfig("CAM_MOUNT_H_deg", 0.0)
CAM_ANGLE_V = pref_category.getFloatConfig("CAM_MOUNT_V_deg", 0.0)
CAM_H = pref_category.getFloatConfig("CAM_H_in", 12.0)
CAM_X = pref_category.getFloatConfig("CAM_X_in", 0.0)
CAM_Y = pref_category.getFloatConfig("CAM_Y_in", 0.0)
GP_H = pref_category.getFloatConfig("GP_H_in", 2.0)

class Detection:
    def __init__(self, r_ground: float, theta_h: float, theta_v: float):
        self.r = r_ground
        self.h_theta = theta_h
        self.v_theta = theta_v

    def getR(self) -> float:
        return self.r
    def getThetaH(self) -> Tuple[float, float]:

        return self.h_theta, self.v_theta
    
    def __str__(self) -> str:
        return f"Game Piece: ({self.r:.2f}, ({self.h_theta:.2f}, {self.v_theta:.2f})"
    def __repr__(self) -> str:
        return self.__str__()

def CALCULATE_PARTIAL_SOLUTION(image: MatLike, boxes) -> List[Detection]:
    global CAM_FOV_X, CAM_FOV_Y, CAM_H, GP_H

    result = []
    for box in boxes:
        
        bounding_box = box.xyxy[0]

        x_avg: float = (bounding_box[0]+bounding_box[2])/2.0
        y_avg: float = (bounding_box[1]+bounding_box[3])/2.0

        x_center: float = image.shape[1] / 2
        y_center: float = image.shape[0] / 2

        x_diff: float = x_avg - x_center
        y_diff: float = y_avg - y_center

        theta_h: float = (x_diff / image.shape[1]) * CAM_FOV_X.valueFloat() + CAM_ANGLE_H.valueFloat()
        theta_v: float = -(y_diff / image.shape[0]) * CAM_FOV_Y.valueFloat() + CAM_ANGLE_V.valueFloat()
        r_ground: float = (GP_H.valueFloat() - CAM_H.valueFloat()) / math.tan(math.radians(theta_v))

        result.append(Detection(r_ground, theta_h, theta_v))

    return result
    
