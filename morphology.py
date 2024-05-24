import cv2


def apply_morphology(image, operation="dilate", kernel_size=(7, 7)):
    if operation == "dilate":
        return cv2.dilate(image, None, iterations=3)
    elif operation == "erode":
        return cv2.erode(image, None, iterations=3)
    elif operation == "open":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
    elif operation == "close":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
    else:
        raise ValueError("Unsupported morphological operation")
