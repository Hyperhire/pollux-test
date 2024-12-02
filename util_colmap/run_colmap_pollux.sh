#!/bin/bash

start_time=$(date +%s)

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --img_path) DATA_PATH="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$DATA_PATH" ]; then
    echo "Error: --img_path parameter is needed."
    exit 1
fi

echo "Data Path : $DATA_PATH"

IMG_PATH="${DATA_PATH}/image"

echo "Image Path : $IMG_PATH"

if [ "$(basename $IMG_PATH)" != "image" ]; then
    echo "Error: The image folder name must be 'image'."
    exit 1
fi

cd ${DATA_PATH}

colmap feature_extractor \
    --image_path $IMG_PATH \
    --database_path database.db \
    --ImageReader.camera_model PINHOLE \
    --SiftExtraction.max_num_features 30000

colmap exhaustive_matcher \
    --database_path database.db

SPARSE_DIR=${DATA_PATH}/sparse
if [ ! -d ${SPARSE_DIR} ]; then
    mkdir -p ${SPARSE_DIR}
fi

colmap mapper \
    --database_path database.db \
    --image_path $IMG_PATH \
    --output_path $DATA_PATH/sparse \
    --Mapper.ba_refine_extra_params=0 \
    --Mapper.ba_refine_focal_length=0 \
    --Mapper.ba_refine_principal_point=0 \
    --Mapper.ba_local_max_refinements=5
# colmap automatic_reconstructor \
#         --workspace_path $DATA_PATH \
#         --image_path $IMG_PATH \
#         --sparse 1 \
#         --dense 0 \
#         --Mapper.ba_refine_extra_params=0 \
#         --Mapper.ba_refine_focal_length=0 \
#         --Mapper.ba_refine_principal_point=0


# python3 /root/workspace/src/util_colmap/bin_to_txt_pollux.py ${SPARSE_DIR}/0

end_time=$(date +%s)
elapsed_time=$(( end_time - start_time ))

echo -e "\e[32mSfM Done!\e[0m"
echo -e "\e[32mTotal elapsed time: $elapsed_time seconds\e[0m"