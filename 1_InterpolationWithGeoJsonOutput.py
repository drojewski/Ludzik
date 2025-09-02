import json
from shapely.geometry import LineString
from pyproj import Transformer
from datetime import datetime

INPUT_FILE = 'export-bemowo.json'         # Twój wejściowy plik z Overpass Turbo!
OUTPUT_FILE = 'export_interpolated.json'  # Wynikowy plik z punktami co 5 m
INTERVAL_METERS = 10           # Interwał co ile metrów generować punkty

# Transformacja: WGS84 → UTM na Warszawę
transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32634", always_xy=True)
transformer_to_wgs84 = Transformer.from_crs("EPSG:32634", "EPSG:4326", always_xy=True)

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    geojson = json.load(f)

features = geojson.get('features', [])
points = []
for idx, feature in enumerate(features):
    geom_type = feature.get('geometry', {}).get('type', None)
    props = feature.get('properties', {})
    if geom_type != 'LineString':
        continue
    coords = feature['geometry']['coordinates']
    if len(coords) < 2:
        continue

    # Przekształć współrzędne do UTM
    coords_utm = [transformer_to_utm.transform(lon, lat) for lon, lat in coords]
    line_utm = LineString(coords_utm)
    total_length = line_utm.length

    distance = 0.0
    while distance <= total_length:
        point_utm = line_utm.interpolate(distance)
        lon, lat = transformer_to_wgs84.transform(point_utm.x, point_utm.y)

        # Dodaj punkt do listy
        points.append({
            "type": "Feature",
            "properties": {
                "segment_id": idx,
                "distance_m": round(distance, 2),
                **props  # zachowaj wszystkie dodatkowe właściwości segmentu
            },
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            }
        })
        distance += INTERVAL_METERS

# GeoJSON z punktami
points_geojson = {
    "type": "FeatureCollection",
    "features": points
}

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(points_geojson, f, ensure_ascii=False, indent=2)

print(f"✅ Wygenerowano {len(points)} punktów co 20m z {len(features)} linii. Plik: {OUTPUT_FILE}")
