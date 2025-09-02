import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

API_KEY = 'AIzaSyDUX8lR6DTTOniAOg3Tk1Xv0Piq0-V0c4A'
PANO_IDS_FILE = 'test_pano_id.txt'
OUTPUT_DIR = 'streetview_images_extended2'
SIZE = '640x640'

HEADINGS = [0, 40, 80, 120, 160, 200, 240, 280, 320]  # 9 azymutów
PITCHES = [-10, 0, 10]                               # 3 pitchy
FOVS = [90, 120]                                     # 2 pola widzenia

MAX_WORKERS = 20  # liczba równoległych wątków, dostosuj według limitów API

os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_image(task):
    pano_id, heading, pitch, fov, idx = task
    url = (
        f"https://maps.googleapis.com/maps/api/streetview?"
        f"size={SIZE}&pano={pano_id}&heading={heading}&pitch={pitch}&fov={fov}&key={API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            filename = f"{OUTPUT_DIR}/{pano_id}_{idx}_h{heading}_p{pitch}_f{fov}.jpg"
            with open(filename, 'wb') as f:
                f.write(resp.content)
            print(f"Pobrano: {filename}")
        else:
            print(f"Błąd pobierania {pano_id} h:{heading} p:{pitch} fov:{fov} - Status: {resp.status_code}")
    except Exception as e:
        print(f"Błąd pobierania {pano_id} h:{heading} p:{pitch} fov:{fov} - {e}")
    sleep(0.05)  # lekkie limitowanie, żeby nie przeciążać sieci

with open(PANO_IDS_FILE, 'r') as f:
    pano_ids = [line.strip().strip('"') for line in f.readlines() if line.strip()]

tasks = []
for pano_id in pano_ids:
    idx = 0
    for heading in HEADINGS:
        for pitch in PITCHES:
            for fov in FOVS:
                tasks.append((pano_id, heading, pitch, fov, idx))
                idx += 1

print(f"Przygotowano {len(tasks)} zadań do pobrania.")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(download_image, task) for task in tasks]
    for _ in as_completed(futures):
        pass

print("Pobieranie zdjęć zakończone.")
