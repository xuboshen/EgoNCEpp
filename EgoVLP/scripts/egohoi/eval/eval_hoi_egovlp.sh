# # single-gpu
# CUDA_VISIBLE_DEVICES=0 python3 run/train_egoclip.py --config configs/eval/egomcq.json
# multi-gpus
HOST_NUM=1
INDEX=0
CHIEF_IP=127.0.0.1
HOST_GPU_NUM=1
torchrun --nproc_per_node=8 run/train_egohoi_lora.py --config configs/eval/egohoi_eval.json