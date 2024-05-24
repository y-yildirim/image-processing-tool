import cv2


def compress_image(image, quality=90):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode(".jpg", image, encode_param)
    if result:
        image = cv2.imdecode(encimg, 1)
    return image
