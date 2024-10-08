# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import collections
import os
import signal
import subprocess
import sys

import transformers
import yaml
from model.scheduler import CosineAnnealingWarmupRestarts
from set_optim_schedule import set_schedule

with open("./EgoNCE_MLM_ITM_Config.yml") as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)


import warnings

import data_loader.data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model_lora as module_arch
import torch
import torch.distributed as dist
import utils.visualizer as module_vis
from parse_config import ConfigParser
from tensorboardX import SummaryWriter
from trainer.trainer_egohoi import Multi_Trainer_dist_hoi
from utils.util import replace_nested_dict_item

warnings.filterwarnings("ignore")


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def main():

    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    # if 'SLURM_JOB_ID' in os.environ:
    #     print("If is being executed")
    #     # single-node and multi-node distributed training on SLURM cluster
    #     # requeue job on SLURM preemption
    #     signal.signal(signal.SIGUSR1, handle_sigusr1)
    #     signal.signal(signal.SIGTERM, handle_sigterm)
    #     # find a common host name on all nodes
    #     # assume scontrol returns hosts in the same order on all nodes
    #     cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
    #     stdout = subprocess.check_output(cmd.split())
    #     host_name = stdout.decode().splitlines()[0]
    #     args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
    #     args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
    #     args.dist_url = f'tcp://{host_name}:58472'
    # else:
    # single-node distributed training
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="tcp://{}:{}".format(args.master_address, args.master_port),
        rank=args.rank,
        world_size=args.world_size,
    )
    rank = dist.get_rank()
    # world_size = dist.get_world_size()
    main_worker(rank, args)
    # torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):

    print("main worker started")
    print("init processed finished")

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    logger = config.get_logger("train")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if config["visualizer"]["type"] != "":
        visualizer = config.initialize(
            name="visualizer",
            module=module_vis,
            exp_name=config["name"],
            web_dir=config._web_log_dir,
        )
    else:
        visualizer = None

    # build tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "./pretrained/" + config["arch"]["args"]["text_params"]["model"],
        TOKENIZERS_PARALLELISM=False,
    )

    # setup data_loader instances
    data_loader, valid_data_loader = init_dataloaders(config, module_data)

    if args.rank == 0:
        print("Train dataset: ", [x.n_samples for x in data_loader], " samples")
        print("Val dataset: ", [x.n_samples for x in valid_data_loader], " samples")

    # build model architecture, then print to console
    model = config.initialize("arch", module_arch)

    if args.rank == 0:
        logger.info(model)

    # get function handles of loss and metrics
    loss = config.initialize(name="loss", module=module_loss)
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    max_steps = int(len(data_loader[0]) * config["trainer"]["epochs"])
    if max_steps == 0:
        max_steps = int(len(data_loader[0]) * 10)
    warmup_steps = config_yaml["warmup_steps"]
    if isinstance(config_yaml["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    optimizer, scheduler = set_schedule(
        model, config, config_yaml, max_steps, warmup_steps
    )

    lr_scheduler = None
    if "lr_scheduler" in config._config:
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=config._config["lr_scheduler"]["args"][
                "first_cycle_steps"
            ]
            // config._config["data_loader"][0]["args"]["batch_size"],
            max_lr=config._config["optimizer"]["args"]["lr"],
            min_lr=config._config["lr_scheduler"]["args"]["min_lr"],
            warmup_steps=config._config["lr_scheduler"]["args"]["warmup_steps"]
            // config._config["data_loader"][0]["args"]["batch_size"],
        )
        print("Use lr_scheduler!")
    writer = None

    if args.rank == 0:
        writer = SummaryWriter(log_dir=str(config.tf_dir))

    print("trainer should being here")
    trainer = Multi_Trainer_dist_hoi(
        args,
        model,
        loss,
        metrics,
        optimizer,
        scheduler,
        gpu,
        config=config,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
        visualizer=visualizer,
        writer=writer,
        tokenizer=tokenizer,
        max_samples_per_epoch=config["trainer"]["max_samples_per_epoch"],
    )

    trainer.train(gpu)


def init_dataloaders(config, module_data):
    """
    We need a way to change split from 'train' to 'val'.
    """
    if "type" in config["data_loader"] and "args" in config["data_loader"]:
        # then its a single dataloader
        data_loader = [config.initialize("data_loader", module_data)]
        config["data_loader"]["args"] = replace_nested_dict_item(
            config["data_loader"]["args"], "split", "val"
        )
        config["data_loader"]["args"] = replace_nested_dict_item(
            config["data_loader"]["args"], "batch_size", 1
        )  # Code is wrong for batch_size != 1
        valid_data_loader = [config.initialize("data_loader", module_data)]
    elif isinstance(config["data_loader"], list):
        data_loader = [
            config.initialize("data_loader", module_data, index=idx)
            for idx in range(len(config["data_loader"]))
        ]
        new_cfg_li = []
        for dl_cfg in config["data_loader"]:
            dl_cfg["args"] = replace_nested_dict_item(dl_cfg["args"], "split", "val")
            # dl_cfg['args'] = replace_nested_dict_item(dl_cfg['args'], 'batch_size', 1) ## Code is wrong for batch_size != 1
            new_cfg_li.append(dl_cfg)
        config._config["data_loader"] = new_cfg_li
        valid_data_loader = [
            config.initialize("data_loader", module_data, index=idx)
            for idx in range(len(config["data_loader"]))
        ]
    else:
        raise ValueError("Check data_loader config, not correct format.")

    return data_loader, valid_data_loader


if __name__ == "__main__":

    try:  # with ddp
        master_address = os.environ["MASTER_ADDR"]
        master_port = int(os.environ["MASTER_PORT"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except Exception as E:  # for debug only
        print(E)
        master_address = 9339
        master_port = 1
        world_size = 1
        rank = 0
        local_rank = 0
    print(master_address, master_port)
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "--task_names", default="EgoNCE_MLM_ITM", type=str, help="Task_Names"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="/fsx/spraman3/Video_Language_Pretraining/Pre-training/EgoVLP_multinode/configs/pt/egoclip.json",
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    parser.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    parser.add_argument(
        "-o", "--observe", action="store_true", help="Whether to observe (neptune)"
    )
    parser.add_argument(
        "-l",
        "--launcher",
        choices=["none", "pytorch"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("-lr1", "--learning_rate1", type=float, default=2e-4)
    parser.add_argument("-sc", "--schedule", default=[60, 80])
    parser.add_argument(
        "--print_freq",
        type=int,
        default=100,
        help="print loss after this number of steps",
    )
    parser.add_argument("--save_dir", type=str, help="dirctory for model saving")
    parser.add_argument("-k", "--local_rank", type=int, default=local_rank)

    parser.add_argument("-ma", "--master_address", default=master_address)
    parser.add_argument("-mp", "--master_port", type=int, default=master_port)
    parser.add_argument("-ws", "--world_size", type=int, default=world_size)
    parser.add_argument("-rk", "--rank", type=int, default=rank)

    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    config = ConfigParser(parser)

    main()
