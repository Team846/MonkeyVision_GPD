import cv2
import numpy as np
from util.config import ConfigCategory, Config
from util.logger import Logger
from cv2.typing import MatLike

pref_category = ConfigCategory("Detection")

min_thresh_win = pref_category.getIntConfig("min_thresh_win", 3)
max_thresh_win = pref_category.getIntConfig("max_thresh_win", 23)
thresh_step = pref_category.getIntConfig("thresh_step", 17)

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
    if IDs is not None:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.aruco.drawDetectedMarkers(image, corners, IDs)
    return image