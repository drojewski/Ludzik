import os
import shutil
import subprocess

source_dir = "lipa"
base_dir = os.path.abspath(".")
output_dir = os.path.join(base_dir, "output")

image_extensions = (".jpg", ".jpeg", ".png")
image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(image_extensions)]

# Znajdź istniejące katalogi numerowane i ustaw startowy numer
existing = [name for name in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, name)) and name.isdigit()]
start_num = max([int(name) for name in existing], default=0) + 1

for i, img_file in enumerate(image_files, start=start_num):
    src_path = os.path.join(source_dir, img_file)
    dest_path = os.path.join(base_dir, "d.jpg")

    # Kopiuj plik jako d.jpg do katalogu bazowego
    shutil.copy2(src_path, dest_path)
    print(f"Kopiuję {src_path} -> {dest_path}")

    # Utwórz katalog output/{i} przed uruchomieniem skryptu
    new_output_subdir = os.path.join(output_dir, str(i))
    os.makedirs(new_output_subdir, exist_ok=True)

    print(f"Uruchamiam 6_MainDino.py dla pliku {img_file}")
    subprocess.run(["python3", "6_MainDino.py"], cwd=base_dir, check=True)

    # Przenieś zawartość katalogu output oprócz podkatalogów numerowanych do output/{i}
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            continue  # pomijamy stare katalogi numerowane
        shutil.move(item_path, new_output_subdir)

    # Przenieś plik d.jpg do katalogu output/{i}
    if os.path.exists(dest_path):
        shutil.move(dest_path, new_output_subdir)
        print(f"Przeniesiono d.jpg do {new_output_subdir}")
    else:
        print("Nie znaleziono pliku d.jpg do przeniesienia")

    # Upewnij się, że katalog output istnieje pusty do kolejnej iteracji
    os.makedirs(output_dir, exist_ok=True)
