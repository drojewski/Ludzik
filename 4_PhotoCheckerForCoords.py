import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = 'AIzaSyDUX8lR6DTTOniAOg3Tk1Xv0Piq0-V0c4A'
INPUT_FILE = 'export_interpolated_plain_array.json'  # Twój plik z listą [lat, lng]
OUTPUT_FILE = 'export_interpolated_plain_array_with_photo.json'
CHECK_URL = 'https://maps.googleapis.com/maps/api/streetview/metadata'
WORKERS = 20

def check_point(args):
    idx, total_points, coord = args
    lat, lng = coord
    location_str = f"{lat},{lng}"
    request_url = f"{CHECK_URL}?location={location_str}&key={API_KEY}"

    log_msg = (
        f"Sprawdzam punkt #{idx}/{total_points}: wsp. {location_str}\n"
        f"   Zapytanie do API: {request_url}"
    )

    try:
        r = requests.get(request_url, timeout=3)
        metadata = r.json()
        has_photo = metadata.get('status', '') == 'OK'
        pano_id = metadata.get('pano_id')
    except Exception as e:
        has_photo = False
        pano_id = None
        log_msg += f"\n   Błąd zapytania: {str(e)}"

    result_msg = (
        f"   --> {'Jest zdjęcie Street View.' if has_photo else 'Brak zdjęcia Street View.'}\n"
        f"   pano_id: {pano_id if pano_id else '[brak pano_id]'}"
    )
    print(log_msg)
    print(result_msg)

    return {
        "coordinates": coord,
        "has_photo": has_photo,
        "pano_id": pano_id,
        "log": log_msg + '\n' + result_msg
    }

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    coords = json.load(f)

total_points = len(coords)
print(f"Do sprawdzenia: {total_points} punktów Street View.\n")

tasks = [
    (idx, total_points, coord)
    for idx, coord in enumerate(coords, 1)
]

results = []
pano_ids_seen = set()
filtered_coords = []

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    future_to_task = {executor.submit(check_point, task): task for task in tasks}
    for future in as_completed(future_to_task):
        result = future.result()
        results.append(result)
        pano_id = result['pano_id']
        if pano_id and pano_id not in pano_ids_seen:
            filtered_coords.append({
                "coordinates": result["coordinates"],
                "pano_id": pano_id
            })
            pano_ids_seen.add(pano_id)
        elif pano_id:
            print(f"Pano_id powtórzony: {pano_id}, punkt pominięty.")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(filtered_coords, f, ensure_ascii=False, indent=2)

print(f"\nGotowe: zapisano {len(filtered_coords)} unikalnych punktów do pliku {OUTPUT_FILE}")
