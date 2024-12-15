
import cv2
from cv2.typing import MatLike
from util.config import ConfigCategory, Config
from camera.preprocess import PROCESS_FRAME, apply_retinex_grayscale
from util.logger import Logger
from time import time_ns
from typing import Tuple

logger = Logger("Camera")
pref_category = ConfigCategory("Camera")

gamma_change_threshold = pref_category.getFloatConfig("gamma_change_threshold", 0.45)
target_gamma_value = pref_category.getFloatConfig("target_gamma_value", 1.0)

def CALCULATE_ADJUSTED_EXPOSURE(gamma, current_exposure) -> int:
    global gamma_change_threshold, target_gamma_value
    if gamma < (target_gamma_value.valueFloat() - gamma_change_threshold.valueFloat()):
        return max(current_exposure - 1, -6)
    elif gamma > (target_gamma_value.valueFloat() + gamma_change_threshold.valueFloat()):
        return min(current_exposure + 1, -1)
    return current_exposure

class CameraReader:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)



        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -4)

        self.camera_exposure = -2

    def get_frame(self) -> Tuple[MatLike, int]:
        _, frame = self.cap.read()
        ts = time_ns()
        frame, gamma = PROCESS_FRAME(frame)
        
        # new_exposure = CALCULATE_ADJUSTED_EXPOSURE(gamma, self.camera_exposure)
        # if new_exposure != self.camera_exposure:
        #     self.camera_exposure = new_exposure

        #     self.cap.set(cv2.CAP_PROP_EXPOSURE, self.camera_exposure)

        return frame, ts