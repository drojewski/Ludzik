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

# Ładowanie modelu z obsługą błędów
try:
    model_name = "vit_large_patch14_dinov2.lvd142m"
    model = timm.create_model(model_name, pretrained=True).to(device)
    model.eval()
    print("Model DINO v2 załadowany pomyślnie")
except Exception as e:
    print(f"Błąd ładowania modelu: {e}")
    exit(1)

def get_dino_embedding(x):
    try:
        with torch.no_grad():
            emb = model(x)
            return emb / emb.norm(dim=-1, keepdim=True)
    except Exception as e:
        print(f"Błąd podczas inference: {e}")
        raise

# Transformacje
transform = transforms.Compose([
    transforms.Resize((518, 518)),  # Zamiast CenterCrop dla szybkości
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Dataset do DataLoadera
class ImageDataset(Dataset):
    def __init__(self, image_files, transform):
        self.image_files = image_files
        self.transform = transform
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_files[idx]).convert("RGB")
            return self.transform(img), self.image_files[idx]
        except (OSError, UnidentifiedImageError) as e:
            print(f"Pominięto {self.image_files[idx]}: {e}")
            # Zwróć pusty tensor/ścieżkę; obsłuż to w pętli
            return None, None

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

def generate_embeddings(batch_size=128, num_workers=4):
    start = time.time()
    all_embs, valid = [], []
    dataset = ImageDataset(image_files, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    total = len(dataset)
    print(f"Batch size: {batch_size}")
    processed = 0
    for imgs, paths in tqdm(loader, total=len(loader), desc='Embeddingowanie'):
        # Usuwaj nieprawidłowe rekordy
        keep = [i for i, p in enumerate(paths) if p is not None]
        if not keep:
            continue
        imgs = imgs[keep]
        valid_paths = [paths[i] for i in keep]
        if not len(valid_paths): continue
        imgs = imgs.to(device)
        try:
            emb = get_dino_embedding(imgs).cpu().numpy()
        except Exception as e:
            print(f"Błąd batcha: {e}")
            continue
        all_embs.append(emb)
        valid.extend(valid_paths)
        processed += len(valid_paths)
        elapsed = time.time() - start
        eta = elapsed / processed * (total - processed)
        print(f"Przetworzono {processed}/{total}, ETA {eta/60:.1f} min")
    if not all_embs:
        print("Brak embeddingów")
        return False
    embs = np.vstack(all_embs).astype('float32')
    faiss.normalize_L2(embs)
    os.makedirs("output", exist_ok=True)
    try:
        np.save("output/embeddings_dino.npy", embs)
        with open("output/image_paths_dino.txt", "w", encoding="utf-8") as f:
            for p in valid: f.write(p + "\n")
        print(f"Zapisano {len(embs)} embeddingów")
        return True
    except Exception as e:
        print(f"Błąd zapisu: {e}")
        return False

# Generowanie lub ładowanie
if not (os.path.exists("output/embeddings_dino.npy") and
        os.path.exists("output/image_paths_dino.txt")):
    print("Generuję embeddingi...")
    if not generate_embeddings(batch_size=128, num_workers=4): exit(1)
else:
    print("Ładuję istniejące embeddingi")
# Ładowanie embeddingów
try:
    embeddings = np.load("output/embeddings_dino.npy")
    with open("output/image_paths_dino.txt", "r", encoding="utf-8") as f:
        image_files = [l.strip() for l in f]
    print(f"Załadowano {len(embeddings)} embeddingów")
except Exception as e:
    print(f"Błąd ładowania: {e}")
    exit(1)
# Indeks FAISS
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
            if len(res) >= k: break
    return res
# Przykład
q="d.jpg"
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

