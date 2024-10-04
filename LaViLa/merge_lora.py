# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
import wandb
from lavila.data import datasets
from lavila.data.video_transforms import Permute
from lavila.models import loss, models
from lavila.models.tokenizer import (
    MyBertTokenizer,
    MyDistilBertTokenizer,
    MyGPT2Tokenizer,
    SimpleTokenizer,
)
from lavila.models.utils import inflate_positional_embeds
from lavila.utils import distributed as dist_utils
from lavila.utils.evaluation_charades import charades_map
from lavila.utils.evaluation_egomcq import (
    egohoi_accuracy_metrics,
    egomcq_accuracy_metrics,
)
from lavila.utils.evaluation_ek100mir import (
    calculate_IDCG,
    calculate_k_counts,
    calculate_mAP,
    calculate_nDCG,
)
from lavila.utils.meter import AverageMeter, ProgressMeter
from lavila.utils.preprocess import generate_label_map
from lavila.utils.random import random_seed
from lavila.utils.scheduler import cosine_scheduler
from torch.distributed.optim import ZeroRedundancyOptimizer


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="lavila finetune and evaluation", add_help=False
    )
    # Data
    parser.add_argument(
        "--dataset",
        default="ek100_mir",
        type=str,
        choices=["ek100_mir", "charades_ego", "ego4d_hoi"],
    )
    parser.add_argument(
        "--root",
        default="datasets/EK100/video_ht256px/",
        type=str,
        help="path to dataset root",
    )
    parser.add_argument("--lora", action="store_true", help="use lora peft")
    parser.add_argument("--lora-r", default=16, type=int, help="lora_r")
    parser.add_argument("--lora-alpha", default=16, type=int, help="lora_alpha")
    parser.add_argument("--lora-dropout", default=0, type=int, help="lora_dropout")
    parser.add_argument(
        "--metadata",
        default="datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_train.csv",
        type=str,
        help="path to metadata file (train set)",
    )
    parser.add_argument(
        "--metadata-val",
        default="datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.csv",
        type=str,
        help="path to metadata file (val set)",
    )
    parser.add_argument(
        "--relevancy-path",
        default="datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl",
        type=str,
        help="path to relevancy matrix (val set)",
    )
    parser.add_argument("--output-dir", default="./", type=str, help="output dir")
    parser.add_argument("--clip-length", default=16, type=int, help="clip length")
    parser.add_argument("--clip-stride", default=4, type=int, help="clip stride")
    parser.add_argument(
        "--sparse-sample", action="store_true", help="switch to sparse sampling"
    )
    # Model
    parser.add_argument(
        "--pretrain-model", default="", type=str, help="path to pretrain model"
    )
    parser.add_argument("--resume", default="", type=str, help="path to resume from")
    parser.add_argument(
        "--find-unused-parameters",
        action="store_true",
        help="do this during DDP (useful for models with tied weights)",
    )
    parser.add_argument(
        "--drop-path-rate", default=0.1, type=float, help="drop path ratio"
    )
    # Training
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--warmup-epochs", default=1, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument(
        "--batch-size",
        default=16,
        type=int,
        help="number of samples per-device/per-gpu",
    )
    parser.add_argument(
        "--max-samples", default=1000000000000, type=int, help="max samples per epoch"
    )
    parser.add_argument(
        "--frozen_text", action="store_true", help="freeze text encoders if set to True"
    )
    parser.add_argument(
        "--freeze-temperature",
        action="store_true",
        help="freeze temperature if set to True",
    )
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument(
        "--fix-lr", action="store_true", help="disable cosine lr decay if set True"
    )
    parser.add_argument(
        "--lr-start", default=1e-6, type=float, help="initial warmup lr"
    )
    parser.add_argument("--lr-end", default=1e-5, type=float, help="minimum final lr")
    parser.add_argument("--clip-grad-type", default="norm", choices=["norm", "value"])
    parser.add_argument("--clip-grad-value", default=None, type=float, help="")
    parser.add_argument(
        "--update-freq",
        default=1,
        type=int,
        help="optimizer update frequency (i.e. gradient accumulation steps)",
    )
    parser.add_argument("--wd", default=0.01, type=float)
    parser.add_argument("--betas", default=(0.9, 0.999), nargs=2, type=float)
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument("--eval-freq", default=5, type=int)
    parser.add_argument("--save-freq", default=5, type=int)
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="disable mixed-precision training (requires more memory and compute)",
    )
    parser.add_argument(
        "--use-zero",
        action="store_true",
        help="use ZeroRedundancyOptimizer to save memory",
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="use gradient checkpointing during training for significantly less GPU usage",
    )
    # System
    parser.add_argument("--print-freq", default=100, type=int, help="print frequency")
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers per process",
    )
    parser.add_argument("--evaluate", action="store_true", help="eval only")
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("--dist-backend", default="nccl", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    return parser


def main(args):
    dist_utils.init_distributed_mode(args)

    global best_acc1
    random_seed(args.seed, dist_utils.get_rank())

    if args.pretrain_model:
        ckpt_path = args.pretrain_model
    else:
        raise Exception("no checkpoint found")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state_dict = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        state_dict[k.replace("module.", "")] = v

    old_args = ckpt["args"]
    print("=> creating model: {}".format(old_args.model))
    try:
        print(old_args.load_visual_pretrained)
    except Exception as E:
        print(E)
        old_args.load_visual_pretrained = None
        old_args.use_cls_token = False
        old_args.project_embed_dim = 256
    model = getattr(models, old_args.model)(
        pretrained=old_args.load_visual_pretrained,
        pretrained2d=old_args.load_visual_pretrained is not None,
        text_use_cls_token=old_args.use_cls_token,
        project_embed_dim=old_args.project_embed_dim,
        timesformer_gated_xattn=False,
        timesformer_freeze_space=False,
        num_frames=args.clip_length,
        drop_path_rate=args.drop_path_rate,
        args=args,
    )

    if args.lora is True:
        model.get_lora()
    model.logit_scale.requires_grad = False
    # import pdb; pdb.set_trace()
    if "TIMESFORMER" in old_args.model or "EGOVLP" in old_args.model:
        # inflate weight
        print("=> inflating PE in models due to different frame numbers")
        state_dict = inflate_positional_embeds(
            model.state_dict(),
            state_dict,
            num_frames=args.clip_length,
            load_temporal_fix="bilinear",
        )
    model.load_state_dict(state_dict, strict=True)
    print(
        "=> loaded resume checkpoint '{}' (epoch {})".format(ckpt_path, ckpt["epoch"])
    )
    model.cuda(args.gpu)
    trainable_params = [sum([v.numel() for k, v in model.named_parameters()])]
    print("tranable_params", trainable_params)

    model.visual = model.visual.merge_and_unload()

    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    optim_params = [
        {"params": p_wd, "weight_decay": args.wd},
        {"params": p_non_wd, "weight_decay": 0},
    ]

    if args.use_zero:
        optimizer = ZeroRedundancyOptimizer(
            optim_params,
            optimizer_class=torch.optim.AdamW,
            lr=args.lr,
            betas=args.betas,
            eps=args.eps,
            weight_decay=args.wd,
        )
    else:
        optimizer = torch.optim.AdamW(
            optim_params,
            lr=args.lr,
            betas=args.betas,
            eps=args.eps,
            weight_decay=args.wd,
        )
    scaler = amp.GradScaler(enabled=not args.disable_amp)

    print("=> saving checkpoint")
    dist_utils.save_on_master(
        {
            "epoch": 0,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_acc1": 0,
            "args": args,
        },
        False,
        args.output_dir,
        is_epoch=6,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "lavila finetune and evaluation", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
