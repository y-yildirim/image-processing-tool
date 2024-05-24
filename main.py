import argparse
import cv2
import os
from compression import compress_image
from noise_reduction import reduce_noise
from edge_detection import detect_edges
from contrast_correction import correct_contrast
from segmentation import segment_image
from thresholding import apply_threshold
from morphology import apply_morphology
from image_repair import repair_image


def parse_arguments():
    parser = argparse.ArgumentParser(description="Image Processing Tool")
    parser.add_argument("input", type=str, help="Path to the input image")
    parser.add_argument("output", type=str, help="Path to save the output image")
    parser.add_argument("--compress", action="store_true", help="Compress the image")
    parser.add_argument(
        "--noise_reduce", action="store_true", help="Reduce noise in the image"
    )
    parser.add_argument("--edge_detect", action="store_true", help="Detect edges in the image")
    parser.add_argument(
        "--contrast", action="store_true", help="Correct contrast of the image"
    )
    parser.add_argument("--segment", action="store_true", help="Segment the image")
    parser.add_argument(
        "--threshold", action="store_true", help="Apply thresholding to the image"
    )
    parser.add_argument(
        "--morphology",
        action="store_true",
        help="Apply morphological operations to the image",
    )
    parser.add_argument("--repair", action="store_true", help="Repair the image")
    return parser.parse_args()


def main():
    args = parse_arguments()
    image = cv2.imread(args.input)

    if args.compress:
        image = compress_image(image)
    if args.noise:
        image = reduce_noise(image)
    if args.edge:
        image = detect_edges(image)
    if args.contrast:
        image = correct_contrast(image)
    if args.segment:
        image = segment_image(image)
    if args.threshold:
        image = apply_threshold(image)
    if args.morphology:
        image = apply_morphology(image)
    if args.repair:
        image = repair_image(image)

    cv2.imwrite(args.output, image)
    print(f"Processed image saved at {args.output}")


if __name__ == "__main__":
    main()
