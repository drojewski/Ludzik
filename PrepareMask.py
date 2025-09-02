import cv2
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# Konfiguracja modelu detectron2 na pretrenowany model do segmentacji instancji
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Próg pewności detekcji
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# Wczytaj obraz
image_path = "to_remove.png"
image = cv2.imread(image_path)

# Predykcja maski
outputs = predictor(image)

# Uzyskaj maski instancji - tensor boolowski [N, H, W]
masks = outputs["instances"].pred_masks.cpu().numpy()
classes = outputs["instances"].pred_classes.cpu().numpy()

# Wybierz klasy do usunięcia, np. osoby (klasa 0 w COCO) i pojazdy (klasy 2 to samochody, 3 to motocykle itp.)
# Możesz dopasować klasy do własnych potrzeb
classes_to_remove = [0, 2, 3, 5, 7]  # 0=człowiek, 2=car, 3=motorcycle, 5=bus, 7=truck

# Inicjalizacja pustej maski
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# Zaznacz na masce obszary wybranych klas
for i, cls in enumerate(classes):
    if cls in classes_to_remove:
        mask = cv2.bitwise_or(mask, masks[i].astype(np.uint8)*255)

# Zapisz maskę - będzie obrazem czarno-białym
cv2.imwrite("mask.png", mask)
