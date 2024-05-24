# torchrun --nproc_per_node=1 --master_port=9998 run/test_charades.py --config configs/eval/charades.json
python run/test_charades.py --config configs/eval/charades_egohoi.json --batch_size 1