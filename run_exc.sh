#!/bin/bash

CUDA_VISIBLE_DEVICES=0 bash run_all_for_image.sh --seq_name our_test_01 --frame_num 150 --save_path ./output/1201_4 --type image
wait $!
CUDA_VISIBLE_DEVICES=0 bash run_all_for_image.sh --seq_name our_test_02 --frame_num 150 --save_path ./output/1201_4 --type image
wait $!
CUDA_VISIBLE_DEVICES=0 bash run_all_for_image.sh --seq_name our_test_03 --frame_num 150 --save_path ./output/1201_4 --type image
wait $!
CUDA_VISIBLE_DEVICES=0 bash run_all_for_image.sh --seq_name our_test_04 --frame_num 150 --save_path ./output/1201_4 --type image
wait $!
CUDA_VISIBLE_DEVICES=0 bash run_all_for_image.sh --seq_name our_test_05 --frame_num 150 --save_path ./output/1201_4 --type image
wait $!
CUDA_VISIBLE_DEVICES=0 bash run_all_for_image.sh --seq_name our_test_06 --frame_num 150 --save_path ./output/1201_4 --type image
wait $!
