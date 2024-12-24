import cv2
import numpy as np
from cv2.typing import MatLike
from util.config import ConfigCategory, Config
from typing import Tuple
from numba import njit

pref_category = ConfigCategory("Preprocessing")

acc_num_bins = pref_category.getIntConfig("acc_num_bins", 1200)
target_brightness = pref_category.getIntConfig("target_brightness", 155)
min_corr_strength = pref_category.getFloatConfig("min_corr_strength", 0.1)
corr_divisor = pref_category.getFloatConfig("corr_divisor", 400.0)
divergence_gain = pref_category.getFloatConfig("divergence_gain", 1.5)

@njit
def COMPUTE_CORRECTION_MATRIX(image: np.ndarray, bins_per_side: int, target_brightness: int) -> np.ndarray:
    height, width = image.shape
    bin_height = height // bins_per_side
    bin_width = width // bins_per_side

    bin_means = np.array([
        np.mean(image[i * bin_height:(i + 1) * bin_height, j * bin_width:(j + 1) * bin_width])
        for i in range(bins_per_side) for j in range(bins_per_side)
    ]).reshape(bins_per_side, bins_per_side)

    correction_matrix = target_brightness - bin_means

    return correction_matrix

def BIN_BASED_CORRECT(image: np.ndarray, acc_num_bins: int, target_brightness: int, min_corr_strength: float, corr_divisor: float) -> np.ndarray:
    height, width = image.shape
    num_bins = acc_num_bins
    bins_per_side = int(np.sqrt(num_bins))

    mean = np.mean(image)
    corr_strength = abs(target_brightness - mean) / corr_divisor + min_corr_strength

    correction_matrix = COMPUTE_CORRECTION_MATRIX(image, bins_per_side, target_brightness)

    correction_matrix = cv2.blur(correction_matrix, (3, 3), 0)

    correction_matrix = cv2.resize(correction_matrix, (width, height), interpolation=cv2.INTER_LINEAR)

    return image.astype(np.float32) + correction_matrix * corr_strength

@njit
def DIVERGING_MOD(image: np.ndarray, divergence_gain: float) -> np.ndarray:
    mean = np.mean(image)

    corrected_image = image * ((divergence_gain * (image - mean) + mean) / mean)
    corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)

    return corrected_image

def PROCESS_FRAME(image: MatLike) -> Tuple[MatLike, float]:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = BIN_BASED_CORRECT(image, acc_num_bins.valueInt(), target_brightness.valueInt(), min_corr_strength.valueFloat(), corr_divisor.valueFloat())
    image = DIVERGING_MOD(image, divergence_gain.valueFloat())

    return image

def GET_DIVERGENCE_GAIN() -> float:
    return divergence_gain.valueFloat()

def SET_DIVERGENCE_GAIN(value: float) -> None:
    divergence_gain.setFloat(value)

def GET_TARGET_BRIGHTNESS() -> int:
    return target_brightness.valueInt()

def SET_TARGET_BRIGHTNESS(value: int) -> None:
    target_brightness.setInt(value)

def GET_NUM_BINS() -> int:
    return acc_num_bins.valueInt()

def SET_NUM_BINS(value: int) -> None:
    acc_num_bins.setInt(value)

def GET_MIN_CORR_STRENGTH() -> float:
    return min_corr_strength.valueFloat()

def SET_MIN_CORR_STRENGTH(value: float) -> None:
    min_corr_strength.setFloat(value)