from typing import List
from networktables import NetworkTables

class NTables:
    def __init__(self, pipeline_number):
        NetworkTables.initialize(server='10.8.46.2')
        self.table = NetworkTables.getTable(f"GPDCam{pipeline_number}")

    def execute(self, detections, latency):
        angles: List[float] = []
        distances: List[float] = []
        tops: List[bool] = []
        for detection in detections:
            angles.append(detection.getTheta())
            distances.append(detection.getR())
            tops.append(detection.isOnTop())
        self.table.putNumberArray("tx", angles)
        self.table.putNumberArray("distances", distances)
        self.table.putBooleanArray("on_tops", tops)
        self.table.putNumber("tl", latency)

