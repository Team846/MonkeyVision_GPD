import cv2
import numpy as np
from util.config import ConfigCategory, Config
from util.logger import Logger
from cv2.typing import MatLike

pref_category = ConfigCategory("Detection")

min_thresh_win = pref_category.getIntConfig("min_thresh_win", 3)
max_thresh_win = pref_category.getIntConfig("max_thresh_win", 23)
thresh_step = pref_category.getIntConfig("thresh_step", 17)

def GET_THRESH_STEP() -> int:
    return thresh_step.valueFloat()

def SET_THRESH_STEP(value: int):
    thresh_step.setInt(value)

def GET_THRESH_WIN() -> int:
    return max_thresh_win.valueFloat()

def SET_THRESH_WIN(value: int):
    max_thresh_win.setInt(value)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

parameters = cv2.aruco.DetectorParameters()

parameters.adaptiveThreshWinSizeMin = min_thresh_win.valueInt()
parameters.adaptiveThreshWinSizeMax = max_thresh_win.valueInt()
parameters.adaptiveThreshWinSizeStep = thresh_step.valueInt()

detector = cv2.aruco.ArucoDetector(dictionary, parameters)

def DETECT_TAGS(image: MatLike):
    global detector
    corners, IDs, _ = detector.detectMarkers(image)
    return (corners, IDs.flatten()) if IDs is not None else (corners, None)

def ANNOTATE_TAGS(image: MatLike, corners, IDs) -> MatLike:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if IDs is not None:
        image = cv2.aruco.drawDetectedMarkers(image, corners, IDs)
    return image