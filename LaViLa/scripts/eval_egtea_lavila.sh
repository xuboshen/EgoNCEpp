CUDA_VISIBLE_DEVICES=1 python eval_zeroshot.py \
 --dataset egtea \
 --metadata-val /fs/fast/base_path/data/egtea/test_split1.txt \
 --root /fs/fast/base_path/data/egtea/cropped_clips/cropped_clips \
 --output-dir /fs/fast/base_path/code/LaViLa/output/zero-shot/egtea/ \
 --batch-size 32 \
 --clip-length 16 \
 --clip-stride 2 \
 --num-clips 10 \
 --num-crops 3 \
 --resume /fs/fast/base_path/code/LaViLa/pretrained/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth \
 --use-half -j 8 \
 --print-freq 10 \
#  --lora
    # --lora \
    # --lora-r 16 \
    # --lora-alpha 16 \
    # --lora-dropout 0 \

