from util.config import ConfigCategory, Config
from util.logger import Logger
from cv2.typing import MatLike
from typing import List, Tuple
import math
import json

logger = Logger("PartialSolution")

pref_category = ConfigCategory(f"PartialSolution")

WD = pref_category.getFloatConfig("WD", 15.5)
ONT = pref_category.getFloatConfig("ONT", 10.0)


def load_calibration(filename: str = "cal.json") -> dict:
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        logger.Log(f"Loaded calibration from {filename}")
        return data
    except Exception as e:
        logger.Error(f"Failed to load calibration file {filename}: {e}")
        return {}


cal = load_calibration()


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
    global cal

    q = cal.get("x", {"a": 0, "b": 0, "c": 0})

    meta = cal.get("meta", {})
    w = meta["resolution"]["width"]

    x_pos *= w / 256.0

    return math.degrees(q["a"] * x_pos + q["b"] * x_pos**3 + q["c"] * x_pos**5)


def vertical_angle(y_pos: float, frame: MatLike) -> float:
    global cal

    q = cal.get("y", {"a": 0, "b": 0, "c": 0})

    meta = cal.get("meta", {})
    h = meta["resolution"]["height"]

    y_pos *= h / 256.0

    return math.degrees(q["a"] * y_pos + q["b"] * y_pos**3 + q["c"] * y_pos**5)


def CALCULATE_PARTIAL_SOLUTION(
    image: MatLike, objs: List[Tuple[float, float]]
) -> List[Detection]:
    global WD, ONT

    result = []

    for obj in objs:
        try:
            objr = [0.0, 0.0, 0.0, 0.0]
            objr[0] = -256 / 2 + obj[0]
            objr[2] = -256 / 2 + obj[2]
            objr[1] = 256 / 2 - obj[1]
            objr[3] = 256 / 2 - obj[3]

            l_h = math.radians(horizontal_angle(objr[0], image))
            r_h = math.radians(horizontal_angle(objr[2], image))

            d_v = math.radians(vertical_angle(objr[1], image))
            u_v = math.radians(vertical_angle(objr[3], image))

            r_ground: float = WD.valueFloat() / (math.tan(r_h) - math.tan(l_h))
            r_ground += WD.valueFloat() / (math.tan(d_v) - math.tan(u_v))
            r_ground /= 2.0

            height: float = r_ground * math.tan((d_v + u_v) / 2.0)

            is_on_top = False

            if height > ONT.valueFloat():
                is_on_top = True

            result.append(Detection(r_ground, math.degrees(l_h + r_h) / 2.0, is_on_top))
        except Exception as e:
            logger.Error(f"Failed to process object {obj}: {e}")

    return result
