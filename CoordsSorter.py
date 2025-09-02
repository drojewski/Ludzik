import json

with open("bemowo_points_20_with_photo_200radius.geojson", "r", encoding="utf-8") as f:
    coords = json.load(f)

sorted_coords = sorted(coords, key=lambda coord: coord[0])

with open("bemowo_points_20_with_photo_200radius_sorted.geojson", "w", encoding="utf-8") as f:
    json.dump(sorted_coords, f, ensure_ascii=False, indent=2)
