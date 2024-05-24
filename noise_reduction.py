import cv2


def reduce_noise(image, method="bilateral", ksize=15):
    if method == "gaussian":
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif method == "median":
        return cv2.medianBlur(image, ksize)
    elif method == "bilateral":
        return cv2.bilateralFilter(image, ksize, 75, 75)
    else:
        raise ValueError("Unsupported noise reduction method")
