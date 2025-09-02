import json
from geopy.distance import geodesic

# pelczynskiego
center = (52.24306717232059, 20.907638686137993)
#center = (52.24306060258832, 20.901662736000546)
radius_meters = 400

with open("export_interpolated_plain_array.json", "r", encoding="utf-8") as f:
    coords = json.load(f)  # lista [ [lng, lat], ... ]

selected_points = []
for lat, lng in coords:
    point = (lat, lng)
    if geodesic(center, point).meters <= radius_meters:
        selected_points.append([lat, lng])

print(f"Znaleziono {len(selected_points)} punktów w promieniu {radius_meters} m.")

with open("export_interpolated_plain_array_clustered_with_photo_ordered.json", "w", encoding="utf-8") as f:
    json.dump(selected_points, f, ensure_ascii=False, indent=2)
