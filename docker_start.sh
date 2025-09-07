#docker run -it --rm \
#  -e DISPLAY=$DISPLAY \
#  -e XAUTHORITY=/tmp/Xauthority \
#  -v /tmp/.X11-unix:/tmp/.X11-unix \
#  -v /run/user/1000/gdm/Xauthority:/tmp/Xauthority:ro \
#  -v /tmp/asd:/app/auto_walk_screenshots \
#  drojewski/geoaga:latest

INPUT_FILE=part_5.json
LOCAL_HOME_DIR=/home/drojewski
LOCAL_APP_DIR=/home/drojewski/GeoAga
LOCAL_SCREENSHOTS_DIR=/home/drojewski/GeoAga/auto_walk_screenshots
LOG_FILE="log_auto_walk_${INPUT_FILE}.txt"

sudo docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v $LOCAL_HOME_DIR/.Xauthority:/tmp/.Xauthority:ro \
  -e XAUTHORITY=/tmp/.Xauthority \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -v $LOCAL_SCREENSHOTS_DIR:/app/auto_walk_screenshots \
  -e INPUT_FILE=$INPUT_FILE \
  -v $LOCAL_APP_DIR/$INPUT_FILE:/app/$INPUT_FILE \
  -v $LOCAL_APP_DIR/$LOG_FILE:/app/$LOG_FILE \
  drojewski/geoaga:latest
