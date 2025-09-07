#docker run -it --rm \
#  -e DISPLAY=$DISPLAY \
#  -e XAUTHORITY=/tmp/Xauthority \
#  -v /tmp/.X11-unix:/tmp/.X11-unix \
#  -v /run/user/1000/gdm/Xauthority:/tmp/Xauthority:ro \
#  -v /tmp/asd:/app/auto_walk_screenshots \
#  drojewski/geoaga:latest

#ustawic nazwe pliku w INPUT_FILE!!
INPUT_FILE=part_5.json
LOG_FILE="log_auto_walk_${INPUT_FILE}.txt"

sudo docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -e XAUTHORITY=/home/drojewski/.Xauthority \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -v /home/drojewski/.Xauthority:/home/drojewski/.Xauthority:ro \
  -v /tmp/auto_walk_screenshots:/app/auto_walk_screenshots \
  -e INPUT_FILE=$INPUT_FILE \
  -v /home/drojewski:/app \
  -v /home/drojewski/$LOG_FILE:/app/$LOG_FILE \
  drojewski/geoaga:latest
