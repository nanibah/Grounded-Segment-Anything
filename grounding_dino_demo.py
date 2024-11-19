from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
import cv2
import os

CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"
DEVICE = "cuda"
IMAGE_FOLDER = "/home/appuser/data/raw/"
IMAGE_PATH = IMAGE_FOLDER + "1943230-206_jpg.rf.a9ff94a8a8714fcc18e86d23b8eddf38.jpg"
RESULT_TXT_FOLDER = "/home/appuser/data/annotations/detection/files"
RESULT_IMG_FOLDER = "/home/appuser/data/annotations/detection/images"
TEXT_PROMPT = "Pallets. Ground."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
FP16_INFERENCE = True

# if FP16_INFERENCE:
#     image = image.half()
#     model = model.half()

# Ensure result folder exists
os.makedirs(RESULT_TXT_FOLDER, exist_ok=True)
os.makedirs(RESULT_IMG_FOLDER, exist_ok=True)

# Load the model
model = load_model(CONFIG_PATH, CHECKPOINT_PATH)

for image_name in os.listdir(IMAGE_FOLDER):
    
    image_path = os.path.join(IMAGE_FOLDER, image_name)
    image_source, image = load_image(image_path)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device=DEVICE,
    )

    result_file_path = os.path.join(RESULT_TXT_FOLDER, f"{os.path.splitext(image_name)[0]}.txt")
    with open(result_file_path, "w") as result_file:
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = box
            label = phrases[i]
            confidence = logits[i]

            if(label == 'ground'):
                calss_id = 0
            elif(label == 'pallets'):
                calss_id = 1

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = abs(x_max - x_min)
            height = abs(y_max - y_min)

            # format required by yolo
            result_file.write(f"{calss_id} {x_center} {y_center} {width} {height}\n")            
            # result_file.write 
            # result_file.write(f"Bounding Box {i}: ({x_min}, {y_min}), ({x_max}, {y_max})\n")
            # result_file.write(f"Label: {label}\n")
            # result_file.write(f"Confidence: {confidence}\n\n")
             

    print(f"Predictions for {image_name} saved to {result_file_path}")
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_image_path = os.path.join(RESULT_IMG_FOLDER, image_name)
    cv2.imwrite(annotated_image_path, annotated_frame)

# # sanity test
# for i, box in enumerate(boxes):
#     x_min, y_min, x_max, y_max = box    # bbx coordinates
#     label = phrases[i]                  # Get the label corresponding to this box
#     confidence = logits[i]              # Get the confidence score for the label
# annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
# cv2.imwrite("annotated_image.jpg", annotated_frame)