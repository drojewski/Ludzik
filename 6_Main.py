import clip
import torch
from PIL import Image
import numpy as np
import faiss
import os
import shutil
from tqdm import tqdm

device = "cpu"  # lub "cuda" jeśli masz GPU

# Załaduj większy model CLIP dla lepszej jakości
model, preprocess = clip.load("ViT-L/14@336px", device=device)

images_dir = "auto_walk_screenshots_2m"
image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.png')]


def generate_embeddings(batch_size=16):  # mniejszy batch dla większego modelu
    all_embeddings = []
    for i in tqdm(range(0, len(image_files), batch_size)):
        batch_files = image_files[i:i + batch_size]
        images = [preprocess(Image.open(f)).to(device) for f in batch_files]
        images_tensor = torch.stack(images)
        with torch.no_grad():
            embeddings = model.encode_image(images_tensor)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)  # normalizacja L2
        all_embeddings.append(embeddings.cpu().numpy())
    all_embeddings = np.vstack(all_embeddings).astype('float32')
    np.save("2m/embeddings_large_2m.npy", all_embeddings)
    with open("2m/image_paths_large.txt", "w") as f:
        for path in image_files:
            f.write(path + "\n")
    print(f"Zapisano {len(all_embeddings)} embeddingów.")


def classify_image_type(image_path):
    """Klasyfikacja semantyczna obrazu"""
    labels = ["blok mieszkalny", "dom jednorodzinny", "ulica miejska", "park", "centrum miasta", "osiedle mieszkaniowe"]
    text_inputs = clip.tokenize(labels).to(device)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(3)  # top 3 kategorie

    results = [(labels[indices[i]], values[i].item()) for i in range(3)]
    return results


# Generuj embeddingi jeśli nie istnieją
if not (os.path.exists("2m/embeddings_large_2m.npy") and os.path.exists("2m/image_paths_large.txt")):
    print("Generowanie embeddingów z modelem ViT-L/14@336px...")
    generate_embeddings()
else:
    print("Ładowanie istniejących embeddingów i ścieżek...")

embeddings = np.load("2m/embeddings_large_2m.npy")
with open("2m/image_paths_large.txt", "r") as f:
    image_files = [line.strip() for line in f.readlines()]

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)


def search_similar(image_path, k=5, threshold=0.25):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        query_embedding = model.encode_image(image)
        query_embedding /= query_embedding.norm(dim=-1, keepdim=True)
    query_np = query_embedding.cpu().numpy().astype('float32')

    # Pobieramy większą pulę kandydatów, by później lepiej wybrać top-k
    distances, indices = index.search(query_np, k * 10)

    print("Top 10 podobieństw:", distances[0][:10])

    results = []
    for dist_arr, idx_arr in zip(distances, indices):
        for dist, idx in zip(dist_arr, idx_arr):
            score = float(dist)
            if score > threshold:
                results.append((image_files[idx], score))
            if len(results) >= k * 3:
                break

    # Wybieramy najlepsze k wyniki
    results = sorted(results, key=lambda x: x[1], reverse=True)[:k]
    return results


def search_similar_with_classification(image_path, k=5, threshold=0.25):
    """Wyszukiwanie z uwzględnieniem klasyfikacji semantycznej"""

    query_categories = classify_image_type(image_path)
    print(f"\nKlasyfikacja obrazu zapytania:")
    for category, confidence in query_categories:
        print(f"  {category}: {confidence:.1%}")

    results = search_similar(image_path, k * 3, threshold)

    enhanced_results = []
    for img_path, similarity_score in results:
        try:
            img_categories = classify_image_type(img_path)

            category_bonus = 0
            for query_cat, query_conf in query_categories[:3]:
                for img_cat, img_conf in img_categories[:3]:
                    if query_cat == img_cat:
                        category_bonus += min(0.15, 0.1 * query_conf * img_conf)

            final_score = similarity_score + category_bonus
            enhanced_results.append((img_path, similarity_score, final_score, img_categories[0]))

        except Exception:
            enhanced_results.append((img_path, similarity_score, similarity_score, ("nieznana", 0)))

    enhanced_results.sort(key=lambda x: x[2], reverse=True)

    return enhanced_results[:k]


os.makedirs("output", exist_ok=True)

query_img = "d.jpg"
print(f"Szukanie podobnych obrazów do: {query_img}")

results = search_similar_with_classification(query_img, k=10, threshold=0.3)

if results:
    print(f"\nZnaleziono {len(results)} podobnych zdjęć:")
    for i, (path, similarity, final_score, (category, cat_conf)) in enumerate(results):
        print(f"{i + 1}. {path}")
        print(f"   Podobieństwo wizualne: {similarity:.4f}")
        print(f"   Wynik końcowy: {final_score:.4f}")
        print(f"   Kategoria: {category} ({cat_conf:.1%})")

        filename = os.path.basename(path)
        output_path = os.path.join("output", f"{i + 1}_{final_score:.3f}_{filename}")
        shutil.copy2(path, output_path)
        print(f"   Skopiowano do: {output_path}\n")
else:
    print("Nie znaleziono zdjęć powyżej progu podobieństwa.")

print("Wszystkie znalezione zdjęcia skopiowano do katalogu 'output'.")
