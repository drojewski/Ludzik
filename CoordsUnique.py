import json

with open("bemowo_points_with_photo_rouned_and_sorted.geojson", "r", encoding="utf-8") as f:
    coords = json.load(f)

# Usuwanie duplikatów
unique_coords = list({tuple(coord) for coord in coords})
unique_coords = [list(coord) for coord in unique_coords]

with open("bemowo_points_with_photo_rouned_and_sorted_unique.geojson", "w", encoding="utf-8") as f:
    json.dump(unique_coords, f, ensure_ascii=False, indent=2)