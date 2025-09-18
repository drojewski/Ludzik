import time
import torch
from PIL import Image, UnidentifiedImageError
import numpy as np
import faiss
import os
import shutil
from tqdm import tqdm
from torchvision import transforms
import timm
from torch.utils.data import DataLoader, Dataset

# Ustawienia i sprawdzenie GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Używane urządzenie: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Ładowanie modelu
try:
    model_name = "vit_large_patch14_dinov2.lvd142m"
    model = timm.create_model(model_name, pretrained=True).to(device)
    model.eval()
    print("Model DINO v2 załadowany pomyślnie")
except Exception as e:
    print(f"Błąd ładowania modelu: {e}")
    exit(1)

def get_dino_embedding(x):
    with torch.no_grad():
        emb = model(x)
        return emb / emb.norm(dim=-1, keepdim=True)

# Transformacje
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
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
            print(f"[SKIP] {path}: {e}")
            return None, None

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return torch.Tensor([]), []
    return torch.utils.data.dataloader.default_collate(batch)

# Lista plików
images_dir = "auto_walk_screenshots"
if not os.path.isdir(images_dir):
    print(f"BŁĄD: Nie znaleziono katalogu {images_dir}")
    exit(1)

image_files = [os.path.join(images_dir, f)
               for f in os.listdir(images_dir)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not image_files:
    print("BŁĄD: Brak plików obrazów")
    exit(1)
print(f"Znaleziono {len(image_files)} obrazów")

# Wznowienie przetwarzania
processed_files = set()
if os.path.exists("output/embeddings_dino.npy") and os.path.exists("output/image_paths_dino.txt"):
    with open("output/image_paths_dino.txt", "r", encoding="utf-8") as f:
        processed_files = {l.strip() for l in f}
    print(f"[DEBUG] Wczytano {len(processed_files)} przetworzonych plików")
    image_files = [f for f in image_files if f not in processed_files]
    print(f"[DEBUG] Pozostało do przetworzenia: {len(image_files)} obrazów")

def save_embeddings(all_embs, valid, processed_files):
    print(f"[DEBUG] save_embeddings: batchy={len(all_embs)}, valid_paths={len(valid)}")
    embs_part = np.vstack(all_embs).astype('float32')
    faiss.normalize_L2(embs_part)
    if os.path.exists("output/embeddings_dino.npy"):
        existing = np.load("output/embeddings_dino.npy")
        embs_part = np.vstack([existing, embs_part])
    all_paths = list(processed_files) + valid
    os.makedirs("output", exist_ok=True)
    np.save("output/embeddings_dino.npy", embs_part)
    with open("output/image_paths_dino.txt", "w", encoding="utf-8") as f:
        for p in all_paths:
            f.write(p + "\n")
    print(f"[DEBUG] Zapisano {len(embs_part)} embeddingów i {len(all_paths)} ścieżek")

def generate_embeddings(batch_size=128, num_workers=4):
    if not image_files:
        print("Brak nowych plików do przetworzenia")
        return True

    start = time.time()
    all_embs, valid = [], []
    dataset = ImageDataset(image_files, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=True, collate_fn=collate_fn)
    total_batches = len(loader)
    print(f"Batch size: {batch_size}, liczba batchy: {total_batches}")
    processed = 0
    save_every = 50

    for i, (imgs, paths) in enumerate(tqdm(loader, desc='Embeddingowanie')):
        print(f"[DEBUG] Start batch {i+1}/{total_batches}")
        if imgs.numel() == 0:
            print("[DEBUG] Batch pusty")
            continue

        imgs = imgs.to(device)
        try:
            emb = get_dino_embedding(imgs).cpu().numpy()
        except Exception as e:
            print(f"[ERROR] Błąd batcha: {e}")
            continue

        all_embs.append(emb)
        valid.extend(paths)
        processed += len(paths)

        elapsed = time.time() - start
        eta = (elapsed / processed * (len(image_files) - processed)) if processed else 0
        print(f"Przetworzono {processed}/{len(image_files)}, ETA {eta/60:.1f} min")

        if (i + 1) % save_every == 0:
            save_embeddings(all_embs, valid, processed_files)
            all_embs.clear()
            valid.clear()

    if all_embs:
        save_embeddings(all_embs, valid, processed_files)
    return True

# Generowanie embeddingów
print("Generuję embeddingi...")
if not generate_embeddings(batch_size=128, num_workers=4):
    exit(1)

print(f"[DEBUG] Pliki w katalogu output/: {os.listdir('output')}")

# Ładowanie embeddingów i ścieżek
try:
    embeddings = np.load("output/embeddings_dino.npy")
    with open("output/image_paths_dino.txt", "r", encoding="utf-8") as f:
        image_files = [l.strip() for l in f]
    print(f"Załadowano {len(embeddings)} embeddingów")
except Exception as e:
    print(f"Błąd ładowania: {e}")
    exit(1)

# Tworzenie indeksu FAISS
try:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
except Exception as e:
    print(f"Błąd tworzenia indeksu: {e}")
    exit(1)

def search_similar(q, k=5, thr=0.25):
    if not os.path.exists(q):
        print(f"Nie znaleziono {q}")
        return []
    try:
        img = Image.open(q).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        qe = get_dino_embedding(tensor).cpu().numpy().astype('float32')
        faiss.normalize_L2(qe)
        D, I = index.search(qe, min(k*10, len(embeddings)))
    except Exception as e:
        print(f"Błąd wyszukiwania: {e}")
        return []
    res = []
    for dist, idx in zip(D[0], I[0]):
        if dist > thr and idx < len(image_files):
            res.append((image_files[idx], float(dist)))
            if len(res) >= k:
                break
    return res

# Przykład wyszukiwania
q = "d.jpg"
print(f"Szukam podobnych do {q}")
out = search_similar(q, k=50, thr=0.5)
if out:
    print(f"Znaleziono {len(out)} wyników")
    for i, (p, s) in enumerate(out, 1):
        print(f"{i}. {p} – {s:.4f}")
        dst = f"output/{i}_{s:.4f}_{os.path.basename(p)}"
        try:
            shutil.copy(p, dst)
            print(f" Skopiowano do {dst}")
        except Exception as e:
            print(f" Błąd kopiowania {p}: {e}")
else:
    print("Brak wyników powyżej progu")
