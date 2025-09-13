import torch
from PIL import Image
import numpy as np
import faiss
import os
import shutil
from tqdm import tqdm
from torchvision import transforms
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"
# Załaduj model DINOv2 (ViT large)
dino_model_name = "vit_large_patch14_dinov2.lvd142m"
model = timm.create_model(dino_model_name, pretrained=True).to(device)
model.eval()


def get_dino_embedding(image_tensor):
    with torch.no_grad():
        emb = model(image_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb


transform = transforms.Compose([
    transforms.Resize(518),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

images_dir = "auto_walk_screenshots"
image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.png')]


def generate_embeddings(batch_size=16):
    all_embeddings = []
    for i in tqdm(range(0, len(image_files), batch_size)):
        batch_files = image_files[i:i + batch_size]
        images = []
        for f in batch_files:
            img = Image.open(f).convert("RGB")
            img = transform(img).to(device)
            images.append(img)
        images_tensor = torch.stack(images)
        embeddings = get_dino_embedding(images_tensor)
        all_embeddings.append(embeddings.cpu().numpy())
    all_embeddings = np.vstack(all_embeddings).astype('float32')
    # L2-normalizacja przed FAISS!
    faiss.normalize_L2(all_embeddings)

    os.makedirs("output", exist_ok=True)  # upewnij się, że katalog output istnieje

    np.save(os.path.join("output", "embeddings_dino.npy"), all_embeddings)
    with open(os.path.join("output", "image_paths_dino.txt"), "w") as f:
        for path in image_files:
            f.write(path + "\n")
    print(f"Zapisano {len(all_embeddings)} embeddingów.")


if not (os.path.exists(os.path.join("output", "embeddings_dino.npy")) and os.path.exists(
        os.path.join("output", "image_paths_dino.txt"))):
    print("Generowanie embeddingów z modelem DINOv2 ViT-L...")
    generate_embeddings()
else:
    print("Ładowanie istniejących embeddingów i ścieżek...")

embeddings = np.load(os.path.join("output", "embeddings_dino.npy"))
with open(os.path.join("output", "image_paths_dino.txt"), "r") as f:
    image_files = [line.strip() for line in f.readlines()]

print("Kształt embeddingów:", embeddings.shape)  # np. (7352, 1024)

# FAISS - poprawny wymiar
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)


def search_similar(image_path, k=5, threshold=0.25):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    query_embedding = get_dino_embedding(img_tensor).cpu().numpy().astype('float32')
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k * 10)
    print("Top 10 podobieństw:", distances[0][:10])
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        score = float(dist)
        if score > threshold:
            results.append((image_files[idx], score))
        if len(results) >= k * 3:
            break
    results = sorted(results, key=lambda x: x[1], reverse=True)[:k]
    return results


os.makedirs("output", exist_ok=True)
query_img = "d.jpg"
print(f"Szukanie podobnych obrazów do: {query_img}")

results = search_similar(query_img, k=50, threshold=0.5)

if results:
    print(f"\nZnaleziono {len(results)} podobnych zdjęć:")
    for i, (path, similarity) in enumerate(results):
        print(f"{i + 1}. {path}")
        print(f"   Podobieństwo wizualne: {similarity:.4f}")
        filename = os.path.basename(path)
        output_path = os.path.join("output", f"{i + 1}_{similarity:.3f}_{filename}")
        shutil.copy2(path, output_path)
        print(f"   Skopiowano do: {output_path}\n")
else:
    print("Nie znaleziono zdjęć powyżej progu podobieństwa.")

print("Wszystkie znalezione zdjęcia skopiowano do katalogu 'output'.")
