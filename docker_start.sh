INPUT_FILE=part_6.json
LOG_FILE="log_auto_walk_${INPUT_FILE}.txt"
XAUTHORITY=/run/user/1000/gdm/Xauthority
touch ./$LOG_FILE
xhost +local:
sudo docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v $XAUTHORITY:/tmp/.Xauthority:ro \
  -e XAUTHORITY=/tmp/.Xauthority \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -v ./auto_walk_screenshots:/app/auto_walk_screenshots \
  -e INPUT_FILE=$INPUT_FILE \
  -v ./$INPUT_FILE:/app/$INPUT_FILE \
  -v ./$LOG_FILE:/app/$LOG_FILE \
  drojewski/geoaga:latest
