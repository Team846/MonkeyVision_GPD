from util.config import ConfigCategory, Config
from ultralytics import YOLO
import cv2
from time import sleep
import asyncio

model = YOLO("algae_ncnn_model", task="detect")

pref_category = ConfigCategory(f"visionY")

CONF = pref_category.getFloatConfig("CONFIDENCE", 0.35)
ASPECT_THRESH = pref_category.getFloatConfig("ASPECT_THRESH", 0.3)


async def runPipeline(frame):
    global CONF, ASPECT_THRESH

    f = cv2.resize(frame, (256, 256))

    boxes = []

    try:
        loop = asyncio.get_running_loop()

        task = loop.run_in_executor(
            None,
            lambda: model.predict(f, conf=CONF.valueFloat(), imgsz=256, verbose=False),
        )

        results = await asyncio.wait_for(task, timeout=0.5)

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

    await asyncio.sleep(0.005)

    return f, boxes
