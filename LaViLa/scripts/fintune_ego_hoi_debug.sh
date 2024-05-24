python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=1234 \
    main_finetune_retrieval.py \
    --dataset ego4d_hoi \
    --metadata /fs/fast/base_path/annotations/egovlpv3/EgoClip_hardnegHOI.csv \
    --metadata-val /fs/fast/base_path/annotations/egovlpv3/EgoMCQ_hardnegHOI.csv \
    --root /fs/fast/base_path/data/ego4d/down_scale/ \
    --output-dir /fs/fast/base_path/code/LaViLa/output/debug \
    --pretrain-model /fs/fast/base_path/code/LaViLa/pretrained/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth \
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