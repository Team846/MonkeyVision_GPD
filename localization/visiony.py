from util.config import ConfigCategory, Config
from ultralytics import YOLO
import cv2
from time import sleep

model = YOLO("algae_ncnn_model", task="detect")

pref_category = ConfigCategory(f"visionY")

CONF = pref_category.getFloatConfig("CONFIDENCE", 0.35)
ASPECT_THRESH = pref_category.getFloatConfig("ASPECT_THRESH", 0.3)


def runPipeline(frame):
    global CONF, ASPECT_THRESH

    f = cv2.resize(frame, (256, 256))

    boxes = []

    try:
        results = model.predict(f, conf=CONF.valueFloat(), imgsz=256, verbose=False)

        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                width = x2 - x1
                height = y2 - y1
                if height == 0:
                    continue
                ratio = width / height
                if (
                    1.0 / (1.0 + ASPECT_THRESH.valueFloat())
                    <= ratio
                    <= 1.0 + ASPECT_THRESH.valueFloat()
                ):
                    boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 0), 1)
    except Exception as e:
        print(e)

    return f, boxes
