from util.config import ConfigCategory, Config
from util.logger import Logger
from cv2.typing import MatLike
from typing import List, Tuple
import math

logger = Logger("PartialSolution")

CAM_FOV_X = 0
CAM_ANGLE_H = 0
PIECE_WIDTH = 0

def SET_CAM(pipeline: int):
    pref_category = ConfigCategory(f"PartialSolution{pipeline}")
    global CAM_FOV_X, CAM_ANGLE_H, PIECE_WIDTH
    CAM_FOV_X = pref_category.getFloatConfig("CAM_FOV_X_deg", 70.0)
    CAM_ANGLE_H = pref_category.getFloatConfig("CAM_MOUNT_H_deg", 0.0)
    PIECE_WIDTH = pref_category.getFloatConfig("PIECE_WIDTH", 17.5)

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
    
def horizontal_angle(x_pos: float, frame: MatLike) -> float:
    global CAM_FOV_X, CAM_ANGLE_H

    x_diff = x_pos - frame.shape[1] / 2.0

    return (x_diff / frame.shape[1]) * CAM_FOV_X.valueFloat() + CAM_ANGLE_H.valueFloat()

def CALCULATE_PARTIAL_SOLUTION(image: MatLike, objs: List[Tuple[float, float]]) -> List[Detection]:
    global CAM_FOV_X, CAM_ANGLE_H, PIECE_WIDTH

    result = []

    for obj in objs:
        theta_h: float = horizontal_angle(obj[0], image)

        l_h = math.radians(horizontal_angle(obj[0] - obj[2], image))
        r_h = math.radians(horizontal_angle(obj[0] + obj[2], image))

        r_ground: float = PIECE_WIDTH.valueFloat() / (math.sin(r_h) - math.sin(l_h))

        result.append(Detection(r_ground, theta_h))

    return result
    
