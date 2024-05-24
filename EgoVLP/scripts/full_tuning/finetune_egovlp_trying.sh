# # single-gpu
# CUDA_VISIBLE_DEVICES=0 python3 run/train_egoclip.py --config configs/eval/egomcq.json
# multi-gpus
HOST_NUM=1
INDEX=0
CHIEF_IP=127.0.0.1
HOST_GPU_NUM=8
torchrun --nproc_per_node=8 /fs/fast/base_path/code/EgoVLP/run/train_egoclip.py --config /fs/fast/base_path/code/EgoVLP/configs/pt/egovlp_lora_hal_try_full_tuning.json