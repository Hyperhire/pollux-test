#!/bin/bash

start_time=$(date +%s)

print_green() {
    echo -e "\033[0;32m$1\033[0m"
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --seq_name) SEQ_NAME="$2"; shift ;;
        --frame_num) FRAME_NUM="$2"; shift ;;
        --save_path) SAVE_PATH="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$SEQ_NAME" ] || [ -z "$FRAME_NUM" ] || [ -z "$SAVE_PATH" ]; then
    echo "Error: Missing required arguments."
    echo "Usage: bash run_all.sh --img_path <img_path> --frame_num \"<frame_num>\""
    exit 1
fi

print_green "SEQ_NAME: $SEQ_NAME"
print_green "FRAME_NUM: $FRAME_NUM"
print_green "SAVE_PATH: $SAVE_PATH"

export PYTHONPATH=$(pwd):$PYTHONPATH

print_green "Running sam_utils.py with --seq_name $SEQ_NAME and --frame_num \"$FRAME_NUM\""
python /root/workspace/src/utils/sam_utils.py --img_path $SEQ_NAME --frame_num "$FRAME_NUM"

# .mp4 제거해줘야 함.
BASENAME="${SEQ_NAME%.*}"
echo $BASENAME

# 다른 폴더에서 진행
RESULT_IMG_PATH="/root/workspace/src/preprocessing/white_background_frames/$BASENAME"
MASK_IMG_PATH="/root/workspace/src/preprocessing/segmented_frames/$BASENAME/binary_mask"
SAVE_DIR="/root/workspace/src/data/$BASENAME/image"
DEST_DIR="/root/workspace/src/data/$BASENAME"
mkdir -p "$SAVE_DIR"
mkdir -p "$DEST_DIR/binary_mask"

cp -r "$RESULT_IMG_PATH/"* "$SAVE_DIR/"
cp -r "$MASK_IMG_PATH/"* "$DEST_DIR/binary_mask/"

print_green "COLMAP_IMG_PATH: $DEST_DIR"

COLMAP_IMG_PATH=$DEST_DIR

print_green "Running run_colmap_pollux.sh with --img_path $COLMAP_IMG_PATH"
bash /root/workspace/src/util_colmap/run_colmap_pollux.sh --img_path $COLMAP_IMG_PATH

print_green "start make normal image"
normal_DIR="/root/workspace/src/data/$BASENAME/normal"
pretrained_DIR='/root/workspace/src/submodules/omnidata/pretrained_models'

if [ -d "$pretrained_DIR" ]; then
  echo "Directory $pretrained_DIR already exist."
else
  sh tools/download_surface_normal_models.sh
fi

# normal_DIR가 존재하는지 확인합니다.
if [ -d "$normal_DIR" ]; then
  echo "Directory $normal_DIR already exist."

else
  echo "Directory $normal_DIR exists."

  # omnidata 하위 폴더로 이동합니다.
  cd /root/workspace/src/submodules/omnidata || { echo "Failed to change directory to /root/workspace/src/submodules/omnidata"; exit 1; }

  # normal estimation을 실행합니다.
  python estimate_normal.py --img_path $SAVE_DIR
fi

end_time=$(date +%s)
elapsed_time=$(( end_time - start_time ))

print_green "All Process Done!"
print_green "Total elapsed time: $elapsed_time seconds"

print_green "Training Start!"

cd /root/workspace/src
python train.py -s /root/workspace/src/data/$BASENAME --config ./config/train.yaml --exp_id $BASENAME --save_path $SAVE_PATH

# TODO
# GPU 선택
# 이미지 개수 딱 안맞음
# Edit --exp_id 
