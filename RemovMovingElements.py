import cv2
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# 1. Konfiguracja Detectron2 Mask R-CNN
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

def remove_dynamic_objects(image_path, output_path):
    img = cv2.imread(image_path)
    outputs = predictor(img)
    masks = outputs["instances"].pred_masks.cpu().numpy()
    classes = outputs["instances"].pred_classes.cpu().numpy()
    dynamic_classes = {0, 2, 3, 5, 7}
    # COCO idx: 0=person,2=car,3=truck,5=bus,7=truck/bus motorcycle etc.

    # Stwórz jednolitą maskę wszystkich dynamicznych obiektów
    dynamic_mask = np.zeros(img.shape[:2], dtype=bool)
    for mask, cls in zip(masks, classes):
        if cls in dynamic_classes:
            dynamic_mask |= mask

    # 2. Maskowanie (czarne tło)
    img[dynamic_mask] = (0, 0, 0)

    # 3. Inpainting (uzupełnianie tła)
    inpainted = cv2.inpaint(img, dynamic_mask.astype('uint8')*255,
                             inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    cv2.imwrite(output_path, inpainted)

# Przykład użycia:
remove_dynamic_objects("auto_walk_screenshots_2m/to_remove.png",
                       "auto_walk_screenshots_2m/point1_cleaned.png")
