docker run --gpus all -it --rm \
  -v /home/drojewski/app/auto_walk_screenshots:/app/auto_walk_screenshots \
  -v /home/drojewski/app/d.jpg:/app/d.jpg \
  -w /app \
  drojewski/geoaga:latest python3 6_MainDino.py
