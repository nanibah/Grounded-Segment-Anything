import os
import cv2
import numpy as np
import supervision as sv
import argparse
import torch
import torchvision
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

print("Starting the script...........................................")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "groundingdino_swint_ogc.pth" #./

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "sam_vit_h_4b8939.pth" #./

# Predict classes and hyper-param for GroundingDINO
SOURCE_IMAGE_FOLDER = "/home/appuser/data/raw/"
SOURCE_IMAGE_PATH = SOURCE_IMAGE_FOLDER + "1943230-206_jpg.rf.a9ff94a8a8714fcc18e86d23b8eddf38.jpg"
OUTPUT_DIR = "/home/appuser/data/annotations/segmentation"
CLASSES = ["The running dog"]
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8

# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


if __name__ == "__main__":   
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    parser = argparse.ArgumentParser(description="Grounded-Segment-Anything Demo")
    parser.add_argument('--config', type=str, default=GROUNDING_DINO_CONFIG_PATH, help='Path to config file')
    parser.add_argument('--grounded_checkpoint', type=str, default=GROUNDING_DINO_CHECKPOINT_PATH, help='Path to grounded model checkpoint')
    parser.add_argument('--input_image', type=str, default=SOURCE_IMAGE_PATH, help='Path to input image')
    parser.add_argument('--text_prompt', type=str, default="ground", help='Text prompt for segmentation')
    parser.add_argument('--output_dir', '-o', default=OUTPUT_DIR, type=str, help='Output directory')
    parser.add_argument('--sam_checkpoint', type=str, default=SAM_CHECKPOINT_PATH, help='SAM checkpoint')
    parser.add_argument('--box_threshold', type=float, default=0.3, help='Box threshold for segmentation')
    parser.add_argument('--text_threshold', type=float, default=0.25, help='Text threshold for segmentation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (e.g., cpu, cuda)')
    args = parser.parse_args()
    
    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    # load image
    image = cv2.imread(SOURCE_IMAGE_PATH)

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _, _ 
        in detections]
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    # save the annotated grounding dino image
    cv2.imwrite("groundingdino_annotated_image.jpg", annotated_frame)

    # NMS post process
    print(f"Before NMS: {len(detections.xyxy)} boxes")
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    print(f"After NMS: {len(detections.xyxy)} boxes")

    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _, _ 
        in detections]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    # save the annotated grounded-sam image
    cv2.imwrite("grounded_sam_annotated_image.jpg", annotated_image)