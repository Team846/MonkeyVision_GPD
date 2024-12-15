
import cv2
from cv2.typing import MatLike
from util.config import ConfigCategory
from camera.preprocess import PROCESS_FRAME
from util.logger import Logger
from time import time_ns
import time
from typing import Tuple

logger = Logger("Camera")

class CameraReader:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080) #TODO: check what it actually is
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -4)

    def get_frame(self) -> Tuple[MatLike, int]:
        _, frame = self.cap.read()
        ts = time_ns()
        frame = PROCESS_FRAME(frame)

        return frame, ts