import argparse
import cv2
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description="Image Processing Tool")
    parser.add_argument("input", type=str, help="Path to the input image")
    parser.add_argument("output_dir", type=str, help="Directory to save the output images")
    parser.add_argument("--compress", action="store_true", help="Compress the image")
    parser.add_argument("--noise", action="store_true", help="Reduce noise in the image")
    parser.add_argument("--edge", action="store_true", help="Detect edges in the image")
    parser.add_argument("--contrast", action="store_true", help="Correct contrast of the image")
    parser.add_argument("--segment", action="store_true", help="Segment the image")
    parser.add_argument("--threshold", action="store_true", help="Apply thresholding to the image")
    parser.add_argument("--morphology", action="store_true", help="Apply morphological operations to the image")
    parser.add_argument("--repair", action="store_true", help="Repair the image")
    parser.add_argument("--mask", type=str, help="Path to the mask image for repair (optional)")
    return parser.parse_args()


def save_image(output_dir, operation, image):
    output_path = os.path.join(output_dir, f"{operation}.jpg")
    cv2.imwrite(output_path, image)
    print(f"{operation.capitalize()} image saved at {output_path}")
