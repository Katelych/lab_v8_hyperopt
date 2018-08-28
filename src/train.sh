#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 \
python main.py \
--data_dir ../data/ \
--summary_dir ../summary/ \
--batch_size 128 \
--learning_rate 0.001 \
--epoch_num 2 
