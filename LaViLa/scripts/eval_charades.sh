export CUDA_VISIBLE_DEVICES=0

python eval_zeroshot.py \
 --dataset charades_ego \
 --metadata-val /fs/fast/base_path/annotations/CharadesEgo/CharadesEgo_v1_test_only1st.csv \
 --root /fs/fast/base_path/data/CharadesEgo/ \
 --output-dir /fs/fast/base_path/code/LaViLa/output/zero-shot/charades_ego/ \
 --batch-size 16 \
 --clip-length 16 \
 --resume $PATH \
 --resume /fs/fast/base_path/code/LaViLa/pretrained/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth \
 --use-half -j 1 \
 --print-freq 10 \
    # --lora \
    # --lora-r 16 \
    # --lora-alpha 16 \
    # --lora-dropout 0 \

