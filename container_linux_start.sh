docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -e XAUTHORITY=/tmp/Xauthority \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /run/user/1000/gdm/Xauthority:/tmp/Xauthority:ro \
  -v /tmp/asd:/app/auto_walk_screenshots \
  drojewski/geoaga:latest
