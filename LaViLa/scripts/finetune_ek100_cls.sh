python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=1234 \
    main_finetune_classification.py \
    --dataset ek100_cls \
    --metadata-train /fs/fast/base_path/data/EK100_256p/epic-kitchens-100-annotations/EPIC_100_train.csv \
    --metadata-val /fs/fast/base_path/data/EK100_256p/epic-kitchens-100-annotations/EPIC_100_validation.csv \
    --root /fs/fast/base_path/data/EK100_256p/ \
    --relevancy-path /fs/fast/base_path/data/EK100_256p/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl \
    --output-dir /fs/fast/base_path/code/LaViLa/output/ek100_cls_bs16 \
    --use-checkpoint \
    --batch-size 16 \
    -j 16 \
    --use-vn-classifier \
    --num-classes 97 300 3806 \
    --use-sgd --wd 4e-5 --lr-multiplier-on-backbone 0.1 \
    --use-checkpoint \
    --pretrain-model /fs/fast/base_path/code/LaViLa/pretrained/merged/checkpoint_0000.pt \
# 
    # --pretrain-model /fs/fast/base_path/code/LaViLa/pretrained/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth \