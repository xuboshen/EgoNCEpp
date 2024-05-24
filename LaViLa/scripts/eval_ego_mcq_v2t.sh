export CUDA_VISIBLE_DEVICES=0

python eval_zeroshot.py \
 --dataset ego4d_mcq_v2t \
 --metadata-val /fs/fast/base_path/annotations/egovlpv3/egohoi.json \
 --root /fs/fast/base_path/data/ego4d/down_scale/ \
 --batch-size 64 \
 --clip-length 4 \
 --resume $PATH \
 --resume /fs/fast/base_path/code/LaViLa/pretrained/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth \
 --use-half -j 4 \
 --print-freq 10
