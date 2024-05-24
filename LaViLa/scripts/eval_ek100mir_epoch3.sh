export CUDA_VISIBLE_DEVICES=1

CUDA_VISIBLE_DEVICES=1 python eval_zeroshot.py \
 --dataset ek100_mir \
 --metadata-val /fs/fast/base_path/data/EK100_256p/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.csv \
 --relevancy-path /fs/fast/base_path/data/EK100_256p/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl \
 --root /fs/fast/base_path/data/EK100_256p/ \
 --output-dir /fs/fast/base_path/code/LaViLa/output/zero-shot/ek100-mir_not_frozen/ \
 --batch-size 64 \
 --clip-length 16 \
 --resume $PATH \
 --resume /fs/fast/base_path/code/LaViLa/output/frozen_token_embed/checkpoint_0003.pt \
 --use-half -j 16 \
 --print-freq 10 \
    --lora \
    --lora-r 16 \
    --lora-alpha 16 \
    --lora-dropout 0 \

