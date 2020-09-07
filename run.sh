#!/bin/bash
BEGIN=$1
END=$2
MODEL=$3
DEVICE=$4

CUDA_VISIBLE_DEVICES=$DEVICE python3 test.py \
	--begin $BEGIN \
	--end $END \
	--model $MODEL \
	--use_rgb \
	--use_balance \
	--use_speed \
	--split_path split_scenes/test_scenes.txt \
	--data_path raw_dataset \
	> "print_logs/"$MODEL"_"$BEGIN"_"$END".txt" 2>&1
