import cv2
import numpy as np


def repair_image(image, method="inpaint", mask=None):
    if mask is None:
        raise ValueError("Mask is required for image repair")
    if method == "inpaint":
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    else:
        raise ValueError("Unsupported repair method")
