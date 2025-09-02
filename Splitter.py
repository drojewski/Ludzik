import json

file_path = 'export_interpolated_plain_array_with_photo.json'  # podaj swoją nazwę pliku z danymi

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

parts = 10
data_len = len(data)
part_size = data_len // parts

for i in range(parts):
    start = i * part_size
    # dla ostatniej części bierz resztę listy
    end = (i + 1) * part_size if i < parts - 1 else data_len
    chunk = data[start:end]
    with open(f'part_{i+1}.json', 'w', encoding='utf-8') as f_out:
        json.dump(chunk, f_out, ensure_ascii=False, indent=2)
