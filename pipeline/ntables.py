from typing import List
from networktables import NetworkTables

class NTables:
    def __init__(self, pipeline_number):
        NetworkTables.initialize(server='10.8.46.2')
        self.table = NetworkTables.getTable(f"GPDCam{pipeline_number}")

    def execute(self, detections, latency):
        angles: List[float] = []
        distances: List[float] = []
        for detection in detections:
            angles.append(detection.getTheta())
            distances.append(detection.getR())
        self.table.putNumberArray("tx", angles)
        self.table.putNumberArray("distances", distances)
        self.table.putNumber("tl", latency)

