from util.config import ConfigCategory, Config
from util.logger import Logger
from cv2.typing import MatLike
from typing import List, Tuple
import math
from interpol import InterpolationTables

logger = Logger("PartialSolution")

IS_TUNING = False

CAM_FOV_X = 0
CAM_FOV_Y = 0
CAM_ANGLE_H = 0
CAM_ANGLE_V = 0
ONT_THRESH = 0


def SET_CAM(pipeline: int):
    pref_category = ConfigCategory(f"PartialSolution{pipeline}")
    global CAM_FOV_X, CAM_ANGLE_H, CAM_FOV_Y, CAM_ANGLE_V, ONT_THRESH
    CAM_FOV_X = pref_category.getFloatConfig("CAM_FOV_X_deg", 71.0)
    CAM_FOV_Y = pref_category.getFloatConfig("CAM_FOV_Y_deg", 55.42)
    CAM_ANGLE_H = pref_category.getFloatConfig("CAM_MOUNT_AH_deg", 0.0)
    CAM_ANGLE_V = pref_category.getFloatConfig("CAM_MOUNT_AV_deg", 0.0)
    ONT_THRESH = pref_category.getFloatConfig("ONT_THRESH", 9.0)


class Detection:
    def __init__(self, r_ground: float, theta_h: float, is_on_top: bool):
        self.r = r_ground
        self.theta = theta_h
        self.is_on_top = is_on_top

    def getR(self) -> float:
        return self.r

    def getTheta(self) -> float:
        return self.theta

    def isOnTop(self) -> bool:
        return self.is_on_top

    def __str__(self) -> str:
        return f"Detection: ({self.r:.2f}, {self.theta:.2f} deg). Top: {self.isOnTop()}"

    def __repr__(self) -> str:
        return self.__str__()


def horizontal_angle(x_pos: float, frame: MatLike) -> float:
    global CAM_FOV_X, CAM_ANGLE_H

    x_diff = x_pos - frame.shape[1] / 2.0

    return (x_diff / frame.shape[1]) * CAM_FOV_X.valueFloat() + CAM_ANGLE_H.valueFloat()


def vertical_angle(y_pos: float, frame: MatLike) -> float:
    global CAM_FOV_Y, CAM_ANGLE_V

    y_diff = y_pos - frame.shape[0] / 2.0

    return (y_diff / frame.shape[0]) * CAM_FOV_Y.valueFloat() + CAM_ANGLE_V.valueFloat()


def CALCULATE_PARTIAL_SOLUTION(
    image: MatLike, objs: List[Tuple[float, float]]
) -> List[Detection]:
    global CAM_FOV_X, CAM_ANGLE_H

    result = []

    for obj in objs:
        theta_h: float = horizontal_angle(obj[0], image)
        theta_v: float = vertical_angle(obj[1], image)

        l_h = math.radians(horizontal_angle(obj[0] - obj[2], image))
        r_h = math.radians(horizontal_angle(obj[0] + obj[2], image))

        r_ground: float = 1.0 / (math.tan(r_h) - math.tan(l_h))
        if not IS_TUNING:
            r_ground = InterpolationTables.interpolate(r_ground)
            r_ground = r_ground / math.cos(math.radians(theta_h))

        height: float = r_ground * math.tan(math.radians(theta_v))

        is_on_top = False

        if height < ONT_THRESH.valueFloat():
            is_on_top = True

        result.append(Detection(r_ground, theta_h, is_on_top))

    return result
