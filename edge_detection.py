import cv2


def detect_edges(image, method="canny", low_threshold=50, high_threshold=150):
    if method == "canny":
        return cv2.Canny(image, low_threshold, high_threshold)
    elif method == "sobel":
        grad_x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
        grad_y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    else:
        raise ValueError("Unsupported edge detection method")
