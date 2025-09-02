import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusionInpaintPipeline

# Ścieżki do plików
input_image_path = "to_remove.png"  # Twój oryginalny obraz
mask_image_path = "mask.png"    # Twoja maska (białe obszary do wypełnienia)

# Wczytaj obraz i maskę jako PIL
image = Image.open(input_image_path).convert("RGB")
mask = Image.open(mask_image_path).convert("L")  # maska w odcieniach szarości

# Ustawienie modelu inpaintingu
model_id = "runwayml/stable-diffusion-inpainting"

pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cpu")  # lub "cpu" jeśli nie masz GPU

# Tekstowy prompt opisujący, co chcesz mieć za wypełnienie (np. "background scenery, natural light")
prompt = "background scenery, natural light"

# Wygeneruj obraz z inpaintingiem
result = pipe(prompt=prompt, image=image, mask_image=mask, guidance_scale=7.5).images[0]

# Zapisz wynik
result.save("output_inpainted.jpg")
print("Wynik zapisany jako output_inpainted.jpg")
