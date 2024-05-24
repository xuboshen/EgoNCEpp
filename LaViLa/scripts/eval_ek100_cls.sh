export CUDA_VISIBLE_DEVICES=0

python eval_zeroshot.py \
 --dataset ek100_cls \
 --metadata-val /fs/fast/base_path/data/EK100_256p/epic-kitchens-100-annotations/validation_close_ov.csv \
 --relevancy-path /fs/fast/base_path/data/EK100_256p/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl \
 --root /fs/fast/base_path/data/EK100_256p/ \
 --output-dir /data1/base_path/code/LaViLa/output/fine-tune/ek100_ov-cls_not_frozen/ \
 --batch-size 64 \
 --resume 'output/ek100_ov/checkpoint_0005.pt' \
 --use-half -j 16 \
 --print-freq 10 \
 --clip-length 16 \
 --cls-use-template

#  --clip-length 16 \
#  --clip-stride 2 \
#  --num-crops 3 --num-clips 10 \