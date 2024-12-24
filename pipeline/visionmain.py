from camera.camerareader import CameraReader
import cv2
import time
import localization.detection
import localization.partial_solution
import pipeline.ntables
from time import time_ns
from typing import List
import platform

class VisionMain:
    def __init__(self, pipeline_number: int):
        self.pipeline_number = pipeline_number

        if platform.system() == "Windows" or platform.system() == "Darwin":
            self.cam = CameraReader(0)
        else:
            self.cam = CameraReader(f"ATCam{pipeline_number}")

        self.frame_count = 0
        self.start_time = time.time()

        self.framerate = 30.0
        self.processing_latency = 0.0

        self.frame: cv2.typing.MatLike = None
        self.detections: List[localization.partial_solution.Detection] = []
        self.ntables : pipeline.ntables.NTables = pipeline.ntables.NTables()

    def execute(self):
        while True:
            frame, timestamp = self.cam.get_frame()

            corners, ids = localization.detection.DETECT_TAGS(frame)

            self.frame = localization.detection.ANNOTATE_TAGS(frame, corners, ids)

            self.detections = localization.partial_solution.CALCULATE_PARTIAL_SOLUTION(frame, corners, ids)

            self.processing_latency = (time_ns() - timestamp) / 1e9

            self.frame_count += 1

            if self.frame_count % 20 == 0:
                end_time = time.time()
                self.framerate = 20 / (end_time - self.start_time)
                self.start_time = end_time
            self.ntables.execute()

    def get_frame(self):
        return self.frame
    
    def get_detections(self):
        return self.detections
    
    def get_framerate(self):
        return self.framerate
    
    def get_processing_latency(self):
        return self.processing_latency
    
    def get_pipeline_number(self):
        return self.pipeline_number