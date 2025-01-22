
import cv2
from cv2.typing import MatLike
# from util.config import ConfigCategory, Config
from util.logger import Logger
from time import time_ns, sleep
from typing import Tuple
import platform

logger = Logger("Camera")

class CameraReader:
    def __init__(self, camera_id):
        if platform.system() == "Windows" or platform.system() == "Darwin":
            self.cap = cv2.VideoCapture(camera_id)
        else:
            self.cap = cv2.VideoCapture(f"/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._{camera_id}_{camera_id}-video-index0")
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FPS, 100.0)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 240)

    def get_frame(self) -> Tuple[MatLike, int]:
        ret, frame = self.cap.read()

        while not ret:
            logger.Warn("Retrying get camera frame...")
            sleep(0.1)
            ret, frame = self.cap.read()

        ts = time_ns()

        return frame, ts