import cv2
from cv2.typing import MatLike
from util.config import ConfigCategory

pref_category = ConfigCategory("Preprocessing")

clahe_clip_limit = pref_category.getFloatConfig("clahe_clip_limit", 2.0)
clahe_tile_grid_size = pref_category.getIntConfig("clahe_tile_grid_size", 8)

clahe = cv2.createCLAHE(
    clipLimit=clahe_clip_limit.valueFloat(),
    tileGridSize=(clahe_tile_grid_size.valueInt(), clahe_tile_grid_size.valueInt()),
)

def APPLY_CLAHE(image: MatLike) -> MatLike:
    global clahe

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    lab[...,0] = clahe.apply(lab[...,0]) #apply CLAHE on L channel

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def PROCESS_FRAME(image: MatLike) -> MatLike:
    image = APPLY_CLAHE(image)

    return image
