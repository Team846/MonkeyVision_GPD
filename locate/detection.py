import cv2
import numpy as np
from util.config import ConfigCategory
from util.logger import Logger
from cv2.typing import MatLike
from ultralytics import YOLO

pref_category = ConfigCategory("Detection")

min_conf = pref_category.getFloatConfig("min_conf", 0.3)
max_iou = pref_category.getFloatConfig("max_iou", 0.5)

model = YOLO('./models/best.onnx')

def DETECT_TAGS(image: MatLike):
    global model

    result = model.predict(image, imgsz=256, conf=min_conf, iou=max_iou)[0]
    Logger.Log("Found {} game pieces!", len(result))
    return result

def ANNOTATE_TAGS(image: MatLike, boxes) -> MatLike:
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image