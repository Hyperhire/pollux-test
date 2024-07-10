#!/bin/bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --seq_name) SEQ_NAME="$2"; shift ;;
        --frame_num) FRAME_NUM="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$SEQ_NAME" ] || [ -z "$FRAME_NUM" ]; then
    echo "Error: Missing required arguments."
    echo "Usage: bash run_all.sh --img_path <img_path> --frame_num \"<frame_num>\""
    exit 1
fi

echo "SEQ_NAME: $SEQ_NAME"
echo "FRAME_NUM: $FRAME_NUM"

export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Running sam_utils.py with --seq_name $SEQ_NAME and --frame_num \"$FRAME_NUM\""
python /root/workspace/src/utils/sam_utils.py --img_path $SEQ_NAME --frame_num "$FRAME_NUM"

$COLMAP_IMG_PATH = "/root/workspace/src/preprocessing/frames_from_video/white_background_frames"/$SEQ_NAME
echo "COLMAP_IMG_PATH: $COLMAP_IMG_PATH"

echo "Running run_colmap_pollux.sh with --img_path $COLMAP_IMG_PATH"
bash /root/workspace/src/util_colmap/run_colmap_pollux.sh --img_path $COLMAP_IMG_PATH