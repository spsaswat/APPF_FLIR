import os
import json
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_image_and_masks(image_path, json_path, max_size=1024):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_height, original_width = image.shape[:2]
        scale = 1.0
        if max(original_height, original_width) > max_size:
            scale = max_size / max(original_height, original_width)
            new_size = (int(original_width * scale), int(original_height * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        with open(json_path, 'r') as f:
            data = json.load(f)

        gt_masks = []
        for shape in data.get('shapes', []):
            if shape.get('label') == 'leaf':
                points = np.array(shape.get('points', []), dtype=np.float32)
                # Adjust points if the image was resized
                if scale != 1.0:
                    points *= scale
                points = points.astype(np.int32)
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [points], 1)
                gt_masks.append(mask.astype(bool))

        return image, gt_masks, image_path
    except Exception as e:
        logger.error(f"Error loading image or masks for {image_path}: {str(e)}")
        return None, None, image_path


def setup_sam_model(checkpoint_path, device):
    logger.info(f"Setting up SAM model with checkpoint: {checkpoint_path}")
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam.to(device=device)
    sam.eval()

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    return mask_generator

def generate_sam_masks(image, mask_generator):
    # Ensure the image has 3 channels (RGB)
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    elif image.shape[-1] > 3:
        image = image[:, :, :3]

    # Resize image so that the longer side is exactly 1024 pixels
    h, w = image.shape[:2]
    if max(h, w) != 1024:
        scale = 1024 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    logger.debug(f"Image shape for generate: {image.shape}, dtype: {image.dtype}")

    # Ensure image is in uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Generate masks
    sam_masks = mask_generator.generate(image)

    return sam_masks

def calculate_metrics(gt_masks, sam_masks, original_shape, resized_shape):
    if not gt_masks or not sam_masks:
        return 0, 0, 0

    # Resize ground truth masks to match SAM output
    resized_gt_masks = []
    for gt_mask in gt_masks:
        resized_gt_mask = cv2.resize(gt_mask.astype(np.uint8), (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_NEAREST)
        resized_gt_masks.append(resized_gt_mask.astype(bool))


    sam_binary_masks = [mask['segmentation'] for mask in sam_masks]

    ious = []
    for resized_gt_mask in resized_gt_masks:
        for sam_mask in sam_binary_masks:
            intersection = np.logical_and(resized_gt_mask, sam_mask).sum()
            union = np.logical_or(resized_gt_mask, sam_mask).sum()
            iou = intersection / union if union > 0 else 0
            ious.append(iou)

    # Calculate metrics
    ap = sum(1 for iou in ious if iou > 0.5) / len(sam_binary_masks) if sam_binary_masks else 0
    ar = sum(1 for iou in ious if iou > 0.5) / len(gt_masks) if gt_masks else 0
    dsc = 2 * ap * ar / (ap + ar) if (ap + ar) > 0 else 0

    return ap, ar, dsc

def main():
    input_folder = 'C:/Users/polis/Downloads/sam_dataset'
    checkpoint_path = 'C:/Users/polis/Downloads/Azure_TL/sam_vit_h_4b8939.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    matched_files = []
    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        json_file = os.path.splitext(img_file)[0] + '.json'
        json_path = os.path.join(input_folder, json_file)

        if os.path.exists(json_path):
            matched_files.append((img_path, json_path))
        else:
            logger.warning(f"No matching JSON file for image: {img_file}")

    logger.info(f"Found {len(matched_files)} matched image-JSON pairs")

    # Initialize the SAM model
    mask_generator = setup_sam_model(checkpoint_path, device)

    # Process images
    results = []
    for img_path, json_path in tqdm(matched_files, desc="Processing images"):
        image, gt_masks, _ = load_image_and_masks(img_path, json_path)
        if image is None:
            continue

        try:
            original_shape = image.shape[:2] # Store original shape
            sam_masks = generate_sam_masks(image, mask_generator)
            resized_shape = image.shape[:2] # Store resized shape
            ap, ar, dsc = calculate_metrics(gt_masks, sam_masks, original_shape, resized_shape) # Pass shapes to calculate_metrics
            results.append((ap, ar, dsc))
            logger.info(f"Metrics for {img_path}: AP={ap:.4f}, AR={ar:.4f}, DSC={dsc:.4f}")
        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")
            continue

    # Calculate average metrics
    if results:
        avg_ap = sum(r[0] for r in results) / len(results)
        avg_ar = sum(r[1] for r in results) / len(results)
        avg_dsc = sum(r[2] for r in results) / len(results)
        logger.info(f"Average metrics: AP={avg_ap:.4f}, AR={avg_ar:.4f}, DSC={avg_dsc:.4f}")
    else:
        logger.warning("No images were successfully processed.")

if __name__ == "__main__":
    main()
