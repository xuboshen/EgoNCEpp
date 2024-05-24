#!/bin/bash
HOST_NUM=1
INDEX=0
CHIEF_IP=127.0.0.1
HOST_GPU_NUM=1

CUDA_VISIBLE_DEVICES=0 torchrun --master_port=9997 --nproc_per_node=1 multinode_train_egohoi.py  --print_freq 100 --config ./configs/eval/egohoi.json --save_dir results/EgoVLPv2_pretrained/EgoHOIBench