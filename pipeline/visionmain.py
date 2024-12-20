from camera.camerareader import CameraReader
import cv2
import time
import locate.detection
import locate.partial_solution
from time import time_ns
from threading import Thread
from typing import List

class VisionMain:
    def __init__(self):
        self.cam = CameraReader()

        self.frame_count = 0
        self.start_time = time.time()

        self.framerate = 30.0
        self.processing_latency = 0.0

        self.frame: cv2.typing.MatLike = None
        self.detections: List[locate.partial_solution.Detection] = []
        time.sleep(5)

    def execute(self):
        while True:
            frame, timestamp = self.cam.get_frame()
            boxes = locate.detection.DETECT_PIECES(frame)

            self.frame = locate.detection.ANNOTATE_PIECES(frame, boxes)

            self.detections = locate.partial_solution.CALCULATE_PARTIAL_SOLUTION(frame, boxes)
            self.processing_latency = (time_ns() - timestamp) / 1e9

            self.frame_count += 1

            if self.frame_count % 20 == 0:
                end_time = time.time()
                self.framerate = 20 / (end_time - self.start_time)
                self.start_time = end_time
            time.sleep(.01)

    def get_frame(self):
        return self.frame
    
    def get_detections(self):
        return self.detections
    
    def get_framerate(self):
        return self.framerate
    
    def get_processing_latency(self):
        return self.processing_latency