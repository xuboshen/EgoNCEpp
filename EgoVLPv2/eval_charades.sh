HOST_NUM=1
INDEX=0
CHIEF_IP=127.0.0.1
HOST_GPU_NUM=1
torchrun --master_port=9994 --nproc_per_node=1 multinode_train_charades.py --save_dir /fs/fast/base_path/code/EgoVLPv2/EgoVLPv2/results/charades/ --config ./configs/eval/charades.json --print_freq 100