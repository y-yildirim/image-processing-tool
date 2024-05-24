import cv2
from compression import compress_image
from noise_reduction import reduce_noise
from edge_detection import detect_edges
from contrast_correction import correct_contrast
from segmentation import segment_image
from thresholding import apply_threshold
from morphology import apply_morphology
from image_repair import repair_image
from utils import parse_arguments, save_image


def main():
    args = parse_arguments()
    image = cv2.imread(args.input)

    if args.compress:
        compressed_image = compress_image(image)
        save_image(args.output_dir, "compress", compressed_image)
    if args.noise:
        noise_reduced_image = reduce_noise(image)
        save_image(args.output_dir, "noise_reduction", noise_reduced_image)
    if args.edge:
        edge_detected_image = detect_edges(image)
        save_image(args.output_dir, "edge_detection", edge_detected_image)
    if args.contrast:
        contrast_corrected_image = correct_contrast(image)
        save_image(args.output_dir, "contrast_correction", contrast_corrected_image)
    if args.segment:
        segmented_image = segment_image(image)
        save_image(args.output_dir, "segmentation", segmented_image)
    if args.threshold:
        thresholded_image = apply_threshold(image)
        save_image(args.output_dir, "thresholding", thresholded_image)
    if args.morphology:
        morphology_image = apply_morphology(image)
        save_image(args.output_dir, "morphology", morphology_image)
    if args.repair:
        mask = cv2.imread(args.mask, 0) if args.mask else None
        repaired_image = repair_image(image, mask=mask)
        save_image(args.output_dir, "repair", repaired_image)


if __name__ == "__main__":
    main()
