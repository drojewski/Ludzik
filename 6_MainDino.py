import os
import shutil
import time
import torch
from PIL import Image, UnidentifiedImageError
import numpy as np
import faiss
from tqdm import tqdm
from torchvision import transforms
import timm
from torch.utils.data import Dataset, DataLoader

# Sprawdzenie dostępnych GPU
print(f"Dostępne GPU: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Używane urządzenie: {device}")

# Ładowanie modelu DINOv2
try:
    model_name = "vit_large_patch14_dinov2.lvd142m"
    model = timm.create_model(model_name, pretrained=True).to(device)
    model.eval()
    print("Model DINO v2 załadowany pomyślnie")
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory // 1024**3
        print(f"GPU memory: {mem} GB")
except Exception as e:
    print(f"Błąd ładowania modelu: {e}")
    exit(1)

# Funkcja do generowania embeddingów
def get_dino_embedding(x):
    with torch.no_grad():
        emb = model(x)
        return emb / emb.norm(dim=-1, keepdim=True)

# Transformacje obrazów
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset z obsługą błędów
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
        except Exception as e:
            print(f"[SKIP] {path}: {e}")
            return None, None

# Collate_fn do DataLoader
def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return torch.tensor([]), []
    imgs, paths = zip(*batch)
    return torch.stack(imgs), list(paths)

# Ścieżki do obrazów
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

# Generowanie embeddingów
def generate_embeddings(batch_size=32):
    dataset = ImageDataset(image_files, transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4,
                        pin_memory=True, collate_fn=collate_fn)
    all_embs = []
    valid_paths = []
    for imgs, paths in tqdm(loader, desc="Embeddingowanie"):
        if imgs.numel() == 0:
            continue
        imgs = imgs.to(device)
        emb = get_dino_embedding(imgs).cpu().numpy()
        all_embs.append(emb)
        valid_paths.extend(paths)
    if not all_embs:
        print("Brak embeddingów do zapisania")
        return False
    all_embs = np.vstack(all_embs).astype('float32')
    faiss.normalize_L2(all_embs)
    os.makedirs("output", exist_ok=True)
    np.save("output/embeddings_dino.npy", all_embs)
    with open("output/image_paths_dino.txt", "w", encoding="utf-8") as f:
        for p in valid_paths:
            f.write(p + "\n")
    print(f"Zapisano {len(valid_paths)} embeddingów")
    return True

# Wygeneruj lub załaduj embeddingi
if not (os.path.exists("output/embeddings_dino.npy") and
        os.path.exists("output/image_paths_dino.txt")):
    print("Generowanie embeddingów...")
    if not generate_embeddings(batch_size=32):
        exit(1)
else:
    print("Ładowanie istniejących embeddingów...")

# Ładowanie embeddingów i ścieżek
embeddings = np.load("output/embeddings_dino.npy")
with open("output/image_paths_dino.txt", "r", encoding="utf-8") as f:
    paths_list = [l.strip() for l in f]
if embeddings.shape[0] != len(paths_list):
    print("BŁĄD: embeddingów i ścieżek się nie zgadza!")
    exit(1)
print(f"Załadowano {len(paths_list)} embeddingów o wymiarze {embeddings.shape[1]}")

# Budowa indeksu FAISS
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Funkcja wyszukiwania podobnych
def search_similar(query_path, k=5, threshold=0.25):
    try:
        img = Image.open(query_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Błąd wczytania {query_path}: {e}")
        return []
    qe = get_dino_embedding(tensor).cpu().numpy().astype('float32')
    faiss.normalize_L2(qe)
    D, I = index.search(qe, min(len(embeddings), k*10))
    results = []
    for dist, idx in zip(D[0], I[0]):
        if dist > threshold:
            results.append((paths_list[idx], float(dist)))
        if len(results) >= k:
            break
    return results

# Przykładowe wyszukiwanie
query_img = "d.jpg"
if os.path.exists(query_img):
    print(f"Szukam podobnych do {query_img}")
    res = search_similar(query_img, k=50, threshold=0.5)
    if res:
        os.makedirs("output", exist_ok=True)
        for i, (p, score) in enumerate(res, 1):
            print(f"{i}. {p} – {score:.4f}")
            dst = f"output/{i}_{score:.4f}_{os.path.basename(p)}"
            shutil.copy2(p, dst)
    else:
        print("Brak wyników powyżej progu")
else:
    print(f"Plik {query_img} nie istnieje")
