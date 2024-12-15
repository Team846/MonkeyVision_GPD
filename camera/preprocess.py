import cv2
import numpy as np
from cv2.typing import MatLike
from util.config import ConfigCategory, Config
from typing import Tuple
import time

pref_category = ConfigCategory("Preprocessing")

clahe_clip_limit = pref_category.getFloatConfig("clahe_clip_limit", 2.0)
clahe_tile_grid_size = pref_category.getIntConfig("clahe_tile_grid_size", 8)

target_brightness = pref_category.getIntConfig("target_brightness", 120)

clahe = cv2.createCLAHE(
    clipLimit=clahe_clip_limit.valueFloat(),
    tileGridSize=(clahe_tile_grid_size.valueInt(), clahe_tile_grid_size.valueInt()),
)

def APPLY_CLAHE(image: MatLike) -> MatLike:
    return clahe.apply(image)


def CALCULATE_GAMMA_CORRECTION(image: np.ndarray) -> float:
    h, w = image.shape[:2]
    step = max(h * w // 100, 1)
    sampled_indices = np.arange(0, h * w, step)[:100]
    sampled_gray = image.reshape(-1)[sampled_indices]

    current_brightness = np.mean(sampled_gray)

    if current_brightness > 0:
        target_brightness_value = target_brightness.valueInt()

        return np.log(current_brightness / 255.0) / np.log(target_brightness_value / 255.0)
    return 1.0


def APPLY_GAMMA_CORRECTION(image: MatLike) -> Tuple[MatLike, float]:
    gamma = CALCULATE_GAMMA_CORRECTION(image)

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")

    image = cv2.LUT(image, table)

    return image, gamma

def PROCESS_FRAME(image: MatLike) -> Tuple[MatLike, float]:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image, gamma = APPLY_GAMMA_CORRECTION(image)

    image = APPLY_CLAHE(image)

    return image, gamma