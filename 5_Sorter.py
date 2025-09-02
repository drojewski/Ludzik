import json
from geopy.distance import geodesic

INPUT_FILE = 'export_interpolated_plain_array_with_photo.json'
OUTPUT_FILE = 'export_interpolated_plain_array_clustered_with_photo_ordered.json'

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    items = json.load(f)

# musisz mieć listę słowników z kluczem 'coordinates'
# struktura: [{'coordinates':[lat,lng], 'pano_id':...}, ...]

# inicjujemy wynik
ordered = []
remaining = items.copy()

# start od pierwszego element
current = remaining.pop(0)
ordered.append(current)

while remaining:
    # wyciągamy lat, lng z obecnego punktu
    current_coords = current['coordinates']
    current_latlng = (current_coords[0], current_coords[1])

    nearest_index = None
    nearest_dist = float('inf')

    for i, point in enumerate(remaining):
        point_coords = point['coordinates']
        pt_latlng = (point_coords[0], point_coords[1])  # poprawiony format

        # oblicz odległość i sprawdź
        dist = geodesic(current_latlng, pt_latlng).meters
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_index = i

    current = remaining.pop(nearest_index)
    ordered.append(current)

# zapis do pliku, gdzie listę `ordered` zawierają słowniki z 'coordinates' i 'pano_id'
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(ordered, f, ensure_ascii=False, indent=2)

print(f"Zapisano {len(ordered)} punktów w kolejności najbliższego sąsiada.")
