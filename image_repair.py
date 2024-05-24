import cv2
import numpy as np


def create_default_mask(image):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    h, w = image.shape[:2]
    mask[h // 4 : h // 4 * 3, w // 4 : w // 4 * 3] = 255
    return mask


def repair_image(image, method="inpaint", mask=None):
    if mask is None:
        mask = create_default_mask(image)
    if method == "inpaint":
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    else:
        raise ValueError("Unsupported repair method")
