export CUDA_VISIBLE_DEVICES=0

python eval_zeroshot.py \
 --dataset ego4d_hoi \
 --metadata-val /fs/fast/base_path/annotations/egovlpv3/actionbench_test.csv \
 --root /fs/fast/base_path/data/ego4d/down_scale/ \
 --batch-size 512 \
 --clip-length 4 \
 --resume $PATH \
 --resume /fs/fast/base_path/code/LaViLa/pretrained/merged/checkpoint_0000.pt \
 --use-half -j 64 \
 --print-freq 10
