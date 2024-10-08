python merge_lora.py \
    --dataset ego4d_hoi \
    --metadata /fs/fast/base_path/annotations/egovlpv3/EgoClip_hardnegHOI.csv \
    --metadata-val /fs/fast/base_path/annotations/egovlpv3/EgoMCQ_hardnegHOI.csv \
    --root /fs/fast/base_path/data/ego4d/down_scale/ \
    --output-dir /fs/fast/base_path/code/LaViLa/pretrained/merged \
    --pretrain-model /fs/fast/base_path/code/LaViLa/output/frozen_token_embed/checkpoint_0006.pt \
    --use-checkpoint \
    --eval-freq 1 \
    --save-freq 1 \
    --batch-size 64 \
    --max-samples 800000 \
    -j 16 \
    --lora \
    --lora-r 16 \
    --lora-alpha 16 \
    --lora-dropout 0 \
    --epochs 10 \
    --clip-length 4\
