import torch
from PIL import Image, UnidentifiedImageError
import numpy as np
import faiss
import os
import shutil
from tqdm import tqdm
from torchvision import transforms
import timm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Sprawdzenie dostępnych GPU
print(f"Dostępne GPU: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Używane urządzenie: {device}")

# Model setup z lepszą obsługą błędów
try:
    dino_model_name = "vit_large_patch14_dinov2.lvd142m"
    model = timm.create_model(dino_model_name, pretrained=True)
    
    # DataParallel dla wykorzystania wszystkich GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Używam DataParallel na {torch.cuda.device_count()} GPU")
    
    model = model.to(device)
    model.eval()
    print("Model załadowany pomyślnie")
    
    # Informacje o pamięci GPU
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory // 1024**3
            print(f"GPU {i} memory: {gpu_memory} GB")
            
except Exception as e:
    print(f"Błąd podczas ładowania modelu: {e}")
    exit(1)

def get_dino_embedding(image_tensor):
    try:
        with torch.no_grad():
            emb = model(image_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb
    except RuntimeError as e:
        print(f"Błąd GPU podczas inference: {e}")
        raise

transform = transforms.Compose([
    transforms.Resize(518),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset class z lepszą obsługą błędów
class ImageDataset(Dataset):
    def __init__(self, image_files, transform):
        self.image_files = image_files
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_files[idx]).convert("RGB")
            img = self.transform(img)
            return img, self.image_files[idx], True
        except Exception as e:  # Szersze łapanie błędów
            print(f"Błąd przetwarzania {self.image_files[idx]}: {e}")
            placeholder = torch.zeros((3, 518, 518))
            return placeholder, self.image_files[idx], False

# Sprawdzenie katalogu obrazów
images_dir = "auto_walk_screenshots"
if not os.path.exists(images_dir):
    print(f"BŁĄD: Katalog {images_dir} nie istnieje!")
    exit(1)

image_files = []
for f in os.listdir(images_dir):
    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_files.append(os.path.join(images_dir, f))

if len(image_files) == 0:
    print(f"BŁĄD: Nie znaleziono obrazów w katalogu {images_dir}")
    exit(1)

print(f"Znaleziono {len(image_files)} plików obrazów")

def generate_embeddings(batch_size=32):
    try:
        dataset = ImageDataset(image_files, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=8, pin_memory=True,
                               drop_last=False)  # Nie porzucaj ostatniego batch-a
        
        all_embeddings = []
        valid_image_files = []
        total_processed = 0
        
        print(f"Rozpoczynam przetwarzanie w batch-ach po {batch_size}")
        
        for batch_imgs, batch_paths, batch_valid in tqdm(dataloader, desc="Generowanie embeddingów"):
            # Filtruj tylko poprawne obrazy
            valid_imgs = []
            valid_paths = []
            
            for img, path, is_valid in zip(batch_imgs, batch_paths, batch_valid):
                if is_valid:
                    valid_imgs.append(img)
                    valid_paths.append(path)
            
            if valid_imgs:
                try:
                    images_tensor = torch.stack(valid_imgs).to(device)
                    embeddings = get_dino_embedding(images_tensor)
                    all_embeddings.append(embeddings.cpu().numpy())
                    valid_image_files.extend(valid_paths)
                    total_processed += len(valid_imgs)
                except Exception as e:
                    print(f"Błąd przetwarzania batch-a: {e}")
                    continue
        
        print(f"Przetworzono {total_processed} poprawnych obrazów")
        
        if all_embeddings:
            try:
                all_embeddings = np.vstack(all_embeddings).astype('float32')
                faiss.normalize_L2(all_embeddings)
                os.makedirs("output", exist_ok=True)
                np.save(os.path.join("output", "embeddings_dino.npy"), all_embeddings)
                with open(os.path.join("output", "image_paths_dino.txt"), "w", encoding='utf-8') as f:
                    for path in valid_image_files:
                        f.write(path + "\n")
                print(f"Zapisano {len(all_embeddings)} embeddingów do pliku")
                return True
            except Exception as e:
                print(f"Błąd zapisu: {e}")
                return False
        else:
            print("Brak poprawnych obrazów do wygenerowania embeddingów.")
            return False
    except Exception as e:
        print(f"Błąd w funkcji generate_embeddings: {e}")
        return False

# Główna logika z lepszą obsługą błędów
try:
    if not (os.path.exists(os.path.join("output", "embeddings_dino.npy")) and
            os.path.exists(os.path.join("output", "image_paths_dino.txt"))):
        print("Generowanie embeddingów z modelem DINOv2 ViT-L...")
        success = generate_embeddings(batch_size=32)
        if not success:
            print("Błąd podczas generowania embeddingów")
            exit(1)
    else:
        print("Ładowanie istniejących embeddingów i ścieżek...")

    # Ładowanie embeddingów z obsługą błędów
    try:
        embeddings = np.load(os.path.join("output", "embeddings_dino.npy"))
        with open(os.path.join("output", "image_paths_dino.txt"), "r", encoding='utf-8') as f:
            image_files_loaded = [line.strip() for line in f.readlines()]
        
        print(f"Kształt embeddingów: {embeddings.shape}")
        print(f"Liczba ścieżek: {len(image_files_loaded)}")
        
        # Sprawdzenie spójności danych
        if len(embeddings) != len(image_files_loaded):
            print("BŁĄD: Liczba embeddingów nie odpowiada liczbie ścieżek!")
            exit(1)
            
    except Exception as e:
        print(f"Błąd ładowania embeddingów: {e}")
        exit(1)
    
    # Tworzenie indeksu FAISS
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    def search_similar(image_path, k=5, threshold=0.25):
        try:
            image = Image.open(image_path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Problem z plikiem zapytania {image_path}: {e}")
            return []
        
        try:
            query_embedding = get_dino_embedding(img_tensor).cpu().numpy().astype('float32')
            faiss.normalize_L2(query_embedding)
            distances, indices = index.search(query_embedding, min(k * 10, len(embeddings)))
            print("Top 10 podobieństw:", distances[0][:10])
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(image_files_loaded):  # Sprawdzenie granic
                    score = float(dist)
                    if score > threshold:
                        results.append((image_files_loaded[idx], score))
                    if len(results) >= k:
                        break
            
            return results
        except Exception as e:
            print(f"Błąd wyszukiwania: {e}")
            return []
    
    # Przykład wyszukiwania - tylko jeśli plik istnieje
    os.makedirs("output", exist_ok=True)
    query_img = "d.jpg"
    
    if os.path.exists(query_img):
        print("Szukanie podobnych obrazów do:", query_img)
        results = search_similar(query_img, k=50, threshold=0.5)
        
        if results:
            print(f"Znaleziono {len(results)} podobnych zdjęć:")
            for i, (path, similarity) in enumerate(results):
                try:
                    print(f"{i + 1}. {path}")
                    print(f"   Podobieństwo wizualne: {similarity:.4f}")
                    filename = os.path.basename(path)
                    output_path = os.path.join("output", f"{i + 1}_{similarity:.3f}_{filename}")
                    shutil.copy2(path, output_path)
                    print(f"   Skopiowano do: {output_path}")
                except Exception as e:
                    print(f"Błąd kopiowania {path}: {e}")
        else:
            print("Nie znaleziono zdjęć powyżej progu podobieństwa.")
    else:
        print(f"Plik zapytania {query_img} nie istnieje - pomijam wyszukiwanie")
    
    print("Zakończono przetwarzanie pomyślnie.")

except Exception as e:
    print(f"Nieprzewidzany błąd: {e}")
    exit(1)

