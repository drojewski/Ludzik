import json

# Wczytanie pliku
with open("bemowo_points_10_with_photo_200m.geojson", "r", encoding="utf-8") as f:
    waypoints = json.load(f)

# Zaokrąglanie i usunięcie duplikatów za pomocą set
rounded_unique = list({(round(lat, 4), round(lng, 4)) for lat, lng in waypoints})

# Jeśli chcesz zapisać wynik z powrotem jako lista list (a nie krotek)
rounded_unique = [list(coord) for coord in rounded_unique]

# Zapis do nowego pliku lub nadpisanie
with open("bemowo_points_with_photo_200m_rounded.geojson", "w", encoding="utf-8") as f:
    json.dump(rounded_unique, f, ensure_ascii=False, indent=2)