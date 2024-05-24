python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=1234 \
    main_finetune_retrieval.py \
    --dataset ek100_mir \
    --metadata /fs/fast/base_path/data/EK100_256p/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_train.csv \
    --metadata-val /fs/fast/base_path/data/EK100_256p/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.csv \
    --root /fs/fast/base_path/data/EK100_256p/ \
    --relevancy-path /fs/fast/base_path/data/EK100_256p/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl \
    --output-dir /fs/fast/base_path/code/LaViLa/output/ek100_mir \
    --pretrain-model /fs/fast/base_path/code/LaViLa/pretrained/merged/checkpoint_0000.pt \
    -j 16 \
    --print-freq 10 \
    --use-checkpoint \
    --batch-size 48
