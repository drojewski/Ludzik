from geopy.distance import geodesic

# Punkt 1: współrzędne w formacie [lat, lng]
punkt1 = [52.24118205740445, 20.908476187259584]

# Punkt 2: współrzędne w formacie [lat, lng]
punkt2 = [52.24124394263725, 20.908717405402275]

# Obliczamy odległość
odl = geodesic(punkt1, punkt2).meters
print(f"Odległość między punktami: {odl:.2f} m")
