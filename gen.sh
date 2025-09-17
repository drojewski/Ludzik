#!/bin/bash

if [ -z "$1" ]; then
  echo "Błąd: Nie podano pliku jako argumentu 1."
  exit 1
fi

if [ -f "$1.done" ]; then
  echo "Dla pliku $1 skrypt był już wykonywany. Plik $1.done istnieje."
  exit 0
fi

if [ ! -f "$HOME/$1" ]; then
  echo "Błąd: Plik $1 nie istnieje w katalogu domowym ($HOME)."
  exit 1
fi

if [ -z "$2" ]; then
  echo "Błąd: Nie podano argumentu 2 (etykieta dla plików wyjściowych)."
  exit 1
fi

rm -rf auto_walk_screenshots
mv "$HOME/$1" .
7z x "$1"
touch "$1.in_progress"
python3 6_MainDino.py
mv output/embeddings_dino.npy output/embeddings_dino-"$2".npy
mv output/image_paths_dino.txt output/image_paths_dino-"$2".txt
mv "$1.in_progress" "$1.done"
rm "$1"
cp -r output/* "$HOME/"
