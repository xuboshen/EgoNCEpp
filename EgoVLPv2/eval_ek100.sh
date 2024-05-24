HOST_NUM=1
INDEX=0
CHIEF_IP=127.0.0.1
HOST_GPU_NUM=1
torchrun --master_port=9997 --nproc_per_node=1 multinode_train_epic.py --save_dir /fs/fast/base_path/code/EgoVLPv2/EgoVLPv2/results/ek100/ --config ./configs/eval/epic.json --print_freq 100