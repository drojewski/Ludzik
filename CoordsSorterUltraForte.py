import json
from geopy.distance import geodesic

def flatten_point(pt):
    while isinstance(pt, list):
        pt = pt
    return pt

with open("bemowo_points_20_with_photo_200radius.geojson", "r", encoding="utf-8") as f:
    coords = json.load(f)

ordered = []
remaining = coords.copy()

current = flatten_point(remaining.pop(0))
ordered.append(current)

while remaining:
    current_latlng = (current[1], current)  # lat, lng

    nearest_index = None
    nearest_dist = float("inf")

    for i, point in enumerate(remaining):
        point = flatten_point(point)

        point_latlng = (point[1], point)
        dist = geodesic(current_latlng, point_latlng).meters

        if dist < nearest_dist:
            nearest_dist = dist
            nearest_index = i

    current = flatten_point(remaining.pop(nearest_index))
    ordered.append(current)

with open("bemowo_points_20_ordered.geojson", "w", encoding="utf-8") as f:
    json.dump(ordered, f, ensure_ascii=False, indent=2)

print("Punkty zostały posortowane przestrzennie i zapisane.")
