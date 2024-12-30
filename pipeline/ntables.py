import cv2
from threading import Thread
from typing import List
from pipeline.visionmain import VisionMain
from networktables import NetworkTables

class NTables:
    def __init__(self, pipeline_number):
        NetworkTables.initialize(server='10.8.46.2')
        self.table = NetworkTables.getTable(f"AprilTagsCam{pipeline_number}")

    def execute(self, detections, latency):
        angles:List[float]=[]
        distances:List[float]=[]
        tags:List[int]=[]
        for detection in detections:
            angles.append(detection.getTheta())
            distances.append(detection.getR())
            tags.append(detection.getTag())
        self.table.putNumberArray("tx", angles)
        self.table.putNumberArray("distances", distances)
        self.table.putNumberArray("tags", tags)
        self.table.putNumber("tl", latency)
