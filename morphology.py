import cv2


def apply_morphology(image, operation="dilate", kernel_size=(5, 5)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    if operation == "dilate":
        return cv2.dilate(image, kernel, iterations=1)
    elif operation == "erode":
        return cv2.erode(image, kernel, iterations=1)
    elif operation == "open":
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == "close":
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError("Unsupported morphological operation")
