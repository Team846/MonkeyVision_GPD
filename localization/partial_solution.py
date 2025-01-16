from util.config import ConfigCategory, Config
from util.logger import Logger
from cv2.typing import MatLike
from typing import List, Tuple
import math

logger = Logger("PartialSolution")

CAM_FOV_X = 0
CAM_FOV_Y = 0
CAM_ANGLE_H = 0
CAM_ANGLE_V = 0
CAM_H = 0
CAM_X = 0
CAM_Y = 0

def SET_CAM(pipeline: int):
    pref_category = ConfigCategory(f"PartialSolution{pipeline}")
    global CAM_FOV_X, CAM_FOV_Y, CAM_ANGLE_H, CAM_ANGLE_V, CAM_H, CAM_X, CAM_Y
    CAM_FOV_X = pref_category.getFloatConfig("CAM_FOV_X_deg", 70.0)
    CAM_FOV_Y = pref_category.getFloatConfig("CAM_FOV_Y_deg", 47.27)
    CAM_ANGLE_H = pref_category.getFloatConfig("CAM_MOUNT_H_deg", 0.0)
    CAM_ANGLE_V = pref_category.getFloatConfig("CAM_MOUNT_V_deg", 0.0)
    CAM_H = pref_category.getFloatConfig("CAM_H_in", 12.0)
    CAM_X = pref_category.getFloatConfig("CAM_X_in", 0.0)
    CAM_Y = pref_category.getFloatConfig("CAM_Y_in", 0.0)

class Detection:
    def __init__(self, r_ground: float, theta_h: float):
        self.r = r_ground
        self.theta = theta_h

    def getR(self) -> float:
        return self.r
    def getTheta(self) -> float:
        return self.theta
    
    def __str__(self) -> str:
        return f"Detection: ({self.r:.2f}, {self.theta:.2f} deg)"
    def __repr__(self) -> str:
        return self.__str__()

def CALCULATE_PARTIAL_SOLUTION(image: MatLike, objs: List[Tuple[float, float]]) -> List[Detection]:
    global CAM_FOV_X, CAM_FOV_Y, CAM_H

    result = []


    for obj in objs:
        x_center: float = image.shape[1] / 2
        y_center: float = image.shape[0] / 2

        x_diff: float = obj[0] - x_center
        y_diff: float = y_center - obj[1]

        theta_h: float = (x_diff / image.shape[1]) * CAM_FOV_X.valueFloat() + CAM_ANGLE_H.valueFloat()
        theta_v: float = (y_diff / image.shape[0]) * CAM_FOV_Y.valueFloat() + CAM_ANGLE_V.valueFloat()
        r_ground: float = (0.0 - CAM_H.valueFloat()) / math.tan(math.radians(theta_v))

        result.append(Detection(r_ground, theta_h))

    return result
    
