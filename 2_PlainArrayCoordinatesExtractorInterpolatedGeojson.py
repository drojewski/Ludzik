import json

# Wczytaj plik geojson
with open("export_interpolated.json", "r", encoding="utf-8") as f:
    data = json.load(f)

coords = []

for feature in data["features"]:
    lon, lat = feature["geometry"]["coordinates"]
    coords.append([lat, lon])

print(coords)

# Zapisz wynik do pliku coords.json
with open("export_interpolated_plain_array.json", "w", encoding="utf-8") as f_out:
    json.dump(coords, f_out, ensure_ascii=False, indent=2)
