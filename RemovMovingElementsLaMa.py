# Najpierw zainstaluj pakiet lama-cleaner (jeśli nie masz)
# pip install lama-cleaner

from lama_cleaner import LaMaCleaner
from PIL import Image
import numpy as np

# Wczytaj obraz wejściowy i maskę (maskę stwórz tak, aby obszary do usunięcia miały biały kolor, reszta czarny)
input_image_path = 'input.jpg'  # Twój obraz oryginalny
mask_image_path = 'mask.png'    # Maska - biały obszar do usunięcia, czarny zostaje

input_image = Image.open(input_image_path).convert("RGB")
mask_image = Image.open(mask_image_path).convert("L")

cleaner = LaMaCleaner()
result = cleaner(input_image, mask_image)

# Zapisz wynik
result.save('output_inpainted.jpg')
