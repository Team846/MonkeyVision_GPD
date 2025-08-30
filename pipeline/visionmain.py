from camera.camerareader import CameraReader
import cv2
import time
import localization.visiony
import localization.partial_solution
import pipeline.ntables
from time import time_ns
from typing import List
import platform
import threading


class VisionMain:
    def __init__(self, pipeline_number: int):
        self.pipeline_number = pipeline_number

        if platform.system() == "Windows" or platform.system() == "Darwin":
            self.cam = CameraReader(pipeline_number)
        else:
            self.cam = CameraReader(f"GPDCam{pipeline_number}")

        self.frame_count = 0
        self.start_time = time.time()

        self.framerate = 30.0
        self.processing_latency = 0.0

        self.frame: cv2.typing.MatLike = None
        self.detections: List[localization.partial_solution.Detection] = []
        self.ntables: pipeline.ntables.NTables = pipeline.ntables.NTables(
            pipeline_number
        )
        self._lock = threading.Lock()

        # localization.partial_solution.SET_CAM(pipeline_number)

    def execute(self):
        frame, timestamp = self.cam.get_frame()

        frame, rawDets = localization.visiony.runPipeline(frame)

        detections = localization.partial_solution.CALCULATE_PARTIAL_SOLUTION(
            frame, rawDets
        )

        processing_latency = (time_ns() - timestamp) / 1e9

        with self._lock:
            self.frame = frame
            self.detections = detections
            self.processing_latency = processing_latency

        self.frame_count += 1

        if self.frame_count % 20 == 0:
            end_time = time.time()
            with self._lock:
                self.framerate = 20 / (end_time - self.start_time)
            self.start_time = end_time
        self.ntables.execute(detections, processing_latency)

    def get_frame(self):
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def get_detections(self):
        with self._lock:
            return self.detections.copy()

    def get_framerate(self):
        with self._lock:
            return self.framerate

    def get_processing_latency(self):
        with self._lock:
            return self.processing_latency

    def get_pipeline_number(self):
        return self.pipeline_number
