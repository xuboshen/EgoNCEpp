#!/bin/bash
HOST_NUM=1
INDEX=0
CHIEF_IP=127.0.0.1
HOST_GPU_NUM=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=9997 --nproc_per_node=8 multinode_train_egohoi.py  --print_freq 100 --config ./configs/pt/egohoi.json --save_dir results/EgoVLPv2_ours/EgoHOIBench --task_name EgoNCE