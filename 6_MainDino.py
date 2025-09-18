import time
import torch
from PIL import Image, UnidentifiedImageError
import numpy as np
import faiss
import os
import shutil
import sys
from tqdm import tqdm
from torchvision import transforms
import timm
from torch.utils.data import DataLoader, Dataset

# Argument paczki
if len(sys.argv) < 2:
    print("ERROR: Nie podano argumentu paczki (np. 1 lub 2_3)")
    sys.exit(1)

batch_label = sys.argv[1]

log_filename = f"embedding_log_{batch_label}.txt"

def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# Ustawienia i sprawdzenie GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"Używane urządzenie: {device}")
if device == "cuda":
    log(f"GPU: {torch.cuda.get_device_name(0)}")

# Ładowanie modelu
try:
    model_name = "vit_large_patch14_dinov2.lvd142m"
    model = timm.create_model(model_name, pretrained=True).to(device)
    model.eval()
    log("Model DINO v2 załadowany pomyślnie")
except Exception as e:
    log(f"Błąd ładowania modelu: {e}")
    sys.exit(1)

def get_dino_embedding(x):
    with torch.no_grad():
        emb = model(x)
        return emb / emb.norm(dim=-1, keepdim=True)

# Transformacje
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset i collate_fn
class ImageDataset(Dataset):
    def __init__(self, files, transform):
        self.files = files
        self.transform = transform
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), path
        except (OSError, UnidentifiedImageError) as e:
            log(f"[SKIP] {path}: {e}")
            return None, None

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return torch.Tensor([]), []
    return torch.utils.data.dataloader.default_collate(batch)

# Lista plików
images_dir = "auto_walk_screenshots"
if not os.path.isdir(images_dir):
    log(f"BŁĄD: Nie znaleziono katalogu {images_dir}")
    sys.exit(1)

image_files = [os.path.join(images_dir, f)
               for f in os.listdir(images_dir)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not image_files:
    log("BŁĄD: Brak plików obrazów")
    sys.exit(1)
log(f"Znaleziono {len(image_files)} obrazów")

# Wznowienie przetwarzania
processed_files = set()
if os.path.exists(f"output/embeddings_dino-{batch_label}.npy") and os.path.exists(f"output/image_paths_dino-{batch_label}.txt"):
    with open(f"output/image_paths_dino-{batch_label}.txt", "r", encoding="utf-8") as f:
        processed_files = {l.strip() for l in f}
    log(f"Wczytano {len(processed_files)} przetworzonych plików")
    image_files = [f for f in image_files if f not in processed_files]
    log(f"Pozostało do przetworzenia: {len(image_files)} obrazów")

def save_embeddings(all_embs, valid, processed_files):
    log(f"Zapisuję batchy: {len(all_embs)}, valid paths: {len(valid)}")
    embs_part = np.vstack(all_embs).astype('float32')
    faiss.normalize_L2(embs_part)
    filename_emb = f"output/embeddings_dino-{batch_label}.npy"
    filename_paths = f"output/image_paths_dino-{batch_label}.txt"
    if os.path.exists(filename_emb):
        existing = np.load(filename_emb)
        embs_part = np.vstack([existing, embs_part])
    
    all_paths = list(processed_files) + valid
    os.makedirs("output", exist_ok=True)
    np.save(filename_emb, embs_part)
    with open(filename_paths, "w", encoding="utf-8") as f:
        for p in all_paths:
            f.write(p + "\n")
    log(f"Zapisano {len(embs_part)} embeddingów i {len(all_paths)} ścieżek")

def generate_embeddings(batch_size=128, num_workers=4):
    if not image_files:
        log("Brak nowych plików do przetworzenia")
        return False
    
    log(f"Rozpoczynam generowanie embeddingów dla: {batch_label}")
    
    start = time.time()
    all_embs, valid = [], []
    dataset = ImageDataset(image_files, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=True, collate_fn=collate_fn)
    total_batches = len(loader)
    log(f"Batch size: {batch_size}, liczba batchy: {total_batches}")
    processed = 0
    save_every = max(1, total_batches // 20)  # zapisz co ~5%
    
    for i, (imgs, paths) in enumerate(tqdm(loader, desc='Embeddingowanie')):
        log(f"Start batch {i+1}/{total_batches}")
        if imgs.numel() == 0:
            log("Batch pusty, pomijam")
            continue
        imgs = imgs.to(device)
        try:
            emb = get_dino_embedding(imgs).cpu().numpy()
        except Exception as e:
            log(f"Błąd batcha: {e}")
            continue
        all_embs.append(emb)
        valid.extend(paths)
        processed += len(paths)
        elapsed = time.time() - start
        eta = elapsed / processed * (len(image_files) - processed)
        log(f"Przetworzono {processed}/{len(image_files)}, ETA {eta/60:.1f} min")
        if (i + 1) % save_every == 0:
            save_embeddings(all_embs, valid, processed_files)
            all_embs.clear()
            valid.clear()
    if all_embs:
        save_embeddings(all_embs, valid, processed_files)
    log(f"Zakończyłem generowanie embeddingów dla: {batch_label}")
    return True

if not generate_embeddings(batch_size=128, num_workers=4):
    log(f"Nie udało się wygenerować embeddingów dla: {batch_label}")
    sys.exit(1)

