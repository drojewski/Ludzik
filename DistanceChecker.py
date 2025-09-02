import json
from geopy.distance import geodesic

with open("bemowo_points_5m_plain_array.json", "r", encoding="utf-8") as f:
    coords = json.load(f)  # lista punktów, najczęściej [[lng, lat], ...]

filtered = []
for point in coords:
    # Dodaj punkt jeśli nie jest za blisko żadnego już wybranego
    if all(geodesic((point[1], point), (other, other)).meters >= 20 for other in filtered):
        filtered.append(point)

print(f"Pozostało {len(filtered)} punktów z {len(coords)}.")
