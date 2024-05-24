export CUDA_VISIBLE_DEVICES=0

python eval_zeroshot.py \
 --dataset charades_ego \
 --metadata-val /fs/fast/base_path/annotations/CharadesEgo/CharadesEgo_v1_test_only1st.csv \
 --root /fs/fast/base_path/data/CharadesEgo/ \
 --output-dir /fs/fast/base_path/code/LaViLa/output/zero-shot/charades_ego/ \
 --batch-size 64 \
 --clip-length 16 \
 --resume $PATH \
 --resume /fs/fast/base_path/code/LaViLa/output/frozen_token_embed/checkpoint_0003.pt \
 --use-half -j 16 \
    --lora \
    --lora-r 16 \
    --lora-alpha 16 \
    --lora-dropout 0 \
 --print-freq 10
