# torchrun --nproc_per_node=1 --master_port=9998 run/test_charades.py --config configs/eval/charades.json
CUDA_VISIBLE_DEVICES=1 python run/test_charades.py --config configs/eval/charades.json