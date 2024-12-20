import cv2
import numpy as np
import torch
from util.config import ConfigCategory
from util.logger import Logger
from cv2.typing import MatLike
from ultralytics import YOLO

pref_category = ConfigCategory("Detection")

logger = Logger("detection")

min_conf = pref_category.getFloatConfig("min_conf", 0.3)
max_iou = pref_category.getFloatConfig("max_iou", 0.5)

# Check for CUDA device and set it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('./models/notes.pt', task="detect").to(device)
if (model.device.type=="cuda"):
    logger.Log(f"Model type is: {model.device.type}")
else:
    logger.Warn(f"Model type is not cuda, but instad {model.device.type}!")


def DETECT_PIECES(image: MatLike):
    global model

    lowerHsv = np.array([0, 200, 200])
    upperHsv = np.array([30, 255, 255])
    hsvImg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsvImg, lowerHsv, upperHsv)
    #Option 1
    if (model.device.type=='cuda'):
        result = model.predict(image, imgsz=256, conf=min_conf.valueFloat(), iou=max_iou.valueFloat(), device=0)[0]
        logger.Log(f"Found {len(result)} game pieces!")
        return result.boxes
    else:
        result = model.predict(image, imgsz=256, conf=min_conf.valueFloat(), iou=max_iou.valueFloat())[0]
        logger.Log(f"Found {len(result)} game pieces!")
        return result.boxes
    
    #Option 2:



def ANNOTATE_PIECES(image: MatLike, boxes) -> MatLike:
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        print((int)(x1))
        image = cv2.rectangle(image, ((int)(x1), (int)(y1)), ((int)(x2), (int)(y2)), (0, 255, 0), 2)
    return image