import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import supervision as sv

def convert_bbox_x1y1x2y2_to_xywh(x1, y1, x2, y2):
    """
    Convert bounding box coordinates from top-left and bottom-right points to 
    top-left point with width and height.
    """
    w = x2 - x1
    h = y2 - y1
    x = x1
    y = y1
    return x, y, w, h

def get_device():
    """
    Get the current device (CPU or CUDA).
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_image_paths(image_dir):
    """
    Get all image paths in a directory.
    """
    return [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]

def segment_image(yolo, mask_predictor, image_path):
    """
    Perform segmentation on an image.
    """
    yolo_output = yolo.predict(image_path, conf=0.5)
    
    r = []
    for result in yolo_output:
        for bbox in result.boxes.data:
            box = bbox.int().cpu().numpy()
            if len(box) >= 6:
                x1, y1, x2, y2, _, cls = box[:6]
                x, y, w, h = convert_bbox_x1y1x2y2_to_xywh(x1, y1, x2, y2)
                r.append([x1, y1, x2, y2, cls])

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_combined = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    output = np.zeros_like(image)

    for i, box in enumerate(r):
        box = box[:-1]
        box = np.array(box)

        mask_predictor.set_image(image)
        masks, scores, logits = mask_predictor.predict(box=box, multimask_output=True)

        detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks)
        detections = detections[detections.area == np.max(detections.area)]

        for m in masks:
            mask_combined = np.logical_or(mask_combined, m)
        
    output[mask_combined] = image[mask_combined]
    
    save_path = f"images/segmented_images/outfit_{os.path.basename(image_path).replace('.jpg', '.png')}"
    cv2.imwrite(save_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

def main():
    """
    Perform segmentation on all images in a directory.
    """
    MODEL_TYPE = "vit_h"
    CHECKPOINT_PATH = os.path.join(os.getcwd(), "models", "sam_weights.pth")
    YOLO_WEIGHTS = r"C:\Users\ANUSHKA NEGI\Pictures\amazon\Future-Fashion-Trends-Forecasting-main\models\yolo_weights.pt"
    IMAGE_DIR = "images/original_images"
    
    if not os.path.exists("images/segmented_images"):
        os.makedirs("images/segmented_images")

    yolo = YOLO(YOLO_WEIGHTS)
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=get_device())
    mask_predictor = SamPredictor(sam)
    
    image_paths = get_image_paths(IMAGE_DIR)
    for image_path in image_paths:
        segmented_image_path = f"images/segmented_images/outfit_{os.path.basename(image_path).replace('.jpg', '.png')}"
        
        if os.path.exists(segmented_image_path):
            print(f"Segmented image {segmented_image_path} already exists, skipping.")
            continue
        
        print(f"Segmenting image: {image_path}")
        segment_image(yolo, mask_predictor, image_path)
        print(f"Segmented image saved to: {segmented_image_path}")

if __name__ == "__main__":
    main()
