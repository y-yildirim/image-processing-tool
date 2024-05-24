import cv2


def apply_threshold(image, method="binary", thresh=127, maxval=255):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == "binary":
        ret, thresh_img = cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY)
    elif method == "otsu":
        ret, thresh_img = cv2.threshold(
            gray, 0, maxval, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif method == "adaptive":
        thresh_img = cv2.adaptiveThreshold(
            gray, maxval, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        raise ValueError("Unsupported thresholding method")
    return thresh_img
