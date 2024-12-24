from util.config import ConfigCategory, Config
from util.logger import Logger
from cv2.typing import MatLike
from typing import List
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

class Detection:
    def __init__(self, r_ground: float, theta_h: float, tagX: float, tagY: float, tag:int):
        self.r = r_ground
        self.theta = theta_h
        self.tagX = tagX
        self.tagY = tagY
        self.tag=tag

    def getTag(self) ->int:
        return self.tag

    def getR(self) -> float:
        return self.r
    def getTheta(self) -> float:
        return self.theta
    def getTagX(self) -> float:
        return self.tagX
    def getTagY(self) -> float:
        return self.tagY
    
    def __str__(self) -> str:
        return f"Detection Tag {self.tag}: ({self.tagX, self.tagY}): ({self.r:.2f}, {self.theta:.2f} deg)"
    def __repr__(self) -> str:
        return self.__str__()

def CALCULATE_PARTIAL_SOLUTION(image: MatLike, all_corners, all_IDs) -> List[Detection]:
    global CAM_FOV_X, CAM_FOV_Y, CAM_H

    result = []

    if all_IDs is None: return result

    for corners, tID in zip(all_corners, all_IDs):
        if str(tID) not in loaded_tags.keys():
            logger.Warn(f"Tag {tID} not found in loaded tags")
            continue

        corners = corners.flatten()
        x_avg: float = (corners[0] + corners[2] + corners[4] + corners[6]) / 4
        y_avg: float = (corners[1] + corners[3] + corners[5] + corners[7]) / 4

        x_center: float = image.shape[1] / 2
        y_center: float = image.shape[0] / 2

        x_diff: float = x_avg - x_center
        y_diff: float = y_avg - y_center

        tag_data = loaded_tags.get(str(tID), {})

        h_tag = float(tag_data.get("h", "54.0"))
        x_tag = float(tag_data.get("x", "0.0"))
        y_tag = float(tag_data.get("y", "0.0"))

        theta_h: float = (x_diff / image.shape[1]) * CAM_FOV_X.valueFloat() + CAM_ANGLE_H.valueFloat()
        theta_v: float = (y_diff / image.shape[0]) * CAM_FOV_Y.valueFloat() + CAM_ANGLE_V.valueFloat()
        r_ground: float = (h_tag - CAM_H.valueFloat()) / math.tan(math.radians(theta_v))

        result.append(Detection(r_ground, theta_h, x_tag, y_tag, tID))

    return result
    
