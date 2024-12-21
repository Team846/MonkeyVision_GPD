
import cv2
from cv2.typing import MatLike
from util.config import ConfigCategory, Config
from camera.preprocess import PROCESS_FRAME
from util.logger import Logger
from time import time_ns, sleep
from typing import Tuple

logger = Logger("Camera")

class CameraReader:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def get_frame(self) -> Tuple[MatLike, int]:
        ret, frame = self.cap.read()

        while not ret:
            logger.Warn("Retrying get camera frame...")
            sleep(0.1)
            ret, frame = self.cap.read()

        ts = time_ns()
        frame = PROCESS_FRAME(frame)

        return frame, ts