export CUDA_VISIBLE_DEVICES=0

python eval_zeroshot.py \
 --dataset charades_ego \
 --metadata-val /fs/fast/base_path/annotations/CharadesEgo/CharadesEgo_v1_test_only1st.csv \
 --root /fs/fast/base_path/data/CharadesEgo/ \
 --output-dir /fs/fast/base_path/code/LaViLa/output/zero-shot/charades_ego/ \
 --batch-size 64 \
 --clip-length 16 \
 --resume $PATH \
 --resume /fs/fast/base_path/code/LaViLa/pretrained/clip_openai_timesformer_base.narrator_rephraser.ep_0001.md5sum_02dbb9.pth \
 --use-half -j 16 \
 --print-freq 10
