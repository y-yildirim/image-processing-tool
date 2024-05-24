import cv2
import numpy as np
from utils import resize_to_match


def create_default_mask(image):
    height, width = image.shape[0], image.shape[1]
    mask = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if image[i, j].sum() > 0:
                mask[i, j] = 0
            else:
                mask[i, j] = 255
    return mask


def repair_image(image, method="inpaint", mask=None):
    if mask is None:
        mask = create_default_mask(image)
    else:
        if mask.shape[:2] != image.shape[:2]:
            mask = resize_to_match(image, mask)
    if method == "inpaint":
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    else:
        raise ValueError("Unsupported repair method")
