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
    transforms.Resize(518),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Lista plików
images_dir = "auto_walk_screenshots"
if not os.path.isdir(images_dir):
    print(f"BŁĄD: Nie znaleziono katalogu {images_dir}")
    exit(1)
image_files = [os.path.join(images_dir,f)
               for f in os.listdir(images_dir)
               if f.lower().endswith(('.png','.jpg','.jpeg'))]
if not image_files:
    print("BŁĄD: Brak plików obrazów")
    exit(1)
print(f"Znaleziono {len(image_files)} obrazów")

def generate_embeddings(batch_size=32):
    start = time.time()
    all_embs, valid = [], []
    total = len(image_files)
    print(f"Batch size: {batch_size}")
    for i in range(0, total, batch_size):
        batch = image_files[i:i+batch_size]
        imgs, paths = [], []
        for p in batch:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(transform(img))
                paths.append(p)
            except (OSError, UnidentifiedImageError) as e:
                print(f"Pominięto {p}: {e}")
        if not imgs: continue
        tensor = torch.stack(imgs).to(device)
        try:
            emb = get_dino_embedding(tensor).cpu().numpy()
        except:
            continue
        all_embs.append(emb)
        valid.extend(paths)
        processed = len(valid)
        elapsed = time.time()-start
        eta = elapsed/processed*(total-processed)
        print(f"Przetworzono {processed}/{total}, ETA {eta/3600:.2f}h")
    if not all_embs:
        print("Brak embeddingów")
        return False
    embs = np.vstack(all_embs).astype('float32')
    faiss.normalize_L2(embs)
    os.makedirs("output", exist_ok=True)
    try:
        np.save("output/embeddings_dino.npy", embs)
        with open("output/image_paths_dino.txt","w",encoding="utf-8") as f:
            for p in valid: f.write(p+"\n")
        print(f"Zapisano {len(embs)} embeddingów")
        return True
    except Exception as e:
        print(f"Błąd zapisu: {e}")
        return False

# Generowanie lub ładowanie
if not (os.path.exists("output/embeddings_dino.npy") and
        os.path.exists("output/image_paths_dino.txt")):
    print("Generuję embeddingi...")
    if not generate_embeddings(batch_size=32): exit(1)
else:
    print("Ładuję istniejące embeddingi")

# Ładowanie embeddingów
try:
    embeddings = np.load("output/embeddings_dino.npy")
    with open("output/image_paths_dino.txt","r",encoding="utf-8") as f:
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
        D,I = index.search(qe, min(k*10,len(embeddings)))
    except Exception as e:
        print(f"Błąd wyszukiwania: {e}")
        return []
    res=[]
    for dist,idx in zip(D[0],I[0]):
        if dist>thr and idx<len(image_files):
            res.append((image_files[idx],float(dist)))
            if len(res)>=k: break
    return res

# Przykład
q="d.jpg"
print(f"Szukam podobnych do {q}")
out = search_similar(q,k=50,thr=0.5)
if out:
    print(f"Znaleziono {len(out)} wyników")
    for i,(p,s) in enumerate(out,1):
        print(f"{i}. {p} – {s:.4f}")
        dst=f"output/{i}_{s:.4f}_{os.path.basename(p)}"
        try: shutil.copy(p,dst); print(f" Skopiowano do {dst}")
        except Exception as e: print(f" Błąd kopiowania {p}: {e}")
else:
    print("Brak wyników powyżej progu")

