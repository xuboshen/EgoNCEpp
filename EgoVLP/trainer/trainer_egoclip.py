import time

import numpy as np
import torch
import torch.distributed as dist
from base import Multi_BaseTrainer_dist
from model.model import sim_matrix
from torch import nn
from tqdm import tqdm
from utils import inf_loop


class AllGather_multi(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, n_gpu, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
            None,
        )


class Multi_Trainer_dist(Multi_BaseTrainer_dist):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(
        self,
        args,
        model,
        loss,
        metrics,
        optimizer,
        config,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
        writer=None,
        visualizer=None,
        tokenizer=None,
        max_samples_per_epoch=50000,
    ):
        super().__init__(args, model, loss, metrics, optimizer, config, writer)
        self.config = config
        self.args = args
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            # take the min
            self.len_epoch = min([len(x) for x in data_loader])
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = self.data_loader[0].batch_size
        self.log_step = int(np.sqrt(self.batch_size))
        self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch
        self.n_gpu = self.args.world_size
        self.allgather = AllGather_multi.apply
        # self.writer = writer

    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
            # if self.writer is not None:
            #     self.writer.log_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _adjust_learning_rate(self, optimizer, epoch, args):
        lr = self.config["optimizer"]["args"]["lr"]
        print(lr)
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """

        self.model.train()
        total_loss = [0] * len(self.data_loader)
        # total_metrics = np.zeros(len(self.metrics))
        start = time.time()
        since = time.time()
        for loader in self.data_loader:
            loader.train_sampler.set_epoch(epoch)
        for batch_idx, data_li in enumerate(zip(*self.data_loader)):
            if (batch_idx + 1) * self.total_batch_sum > self.max_samples_per_epoch:
                break
            for dl_idx, data in enumerate(data_li):
                # then assume we must tokenize the input, e.g. its a string
                if "text_neg" in data.keys():  # w/ negative sampling
                    text_neg = []
                    for neg in data["text_neg"]:
                        text_neg.extend(neg.split(","))
                    data["text"] = data["text"] + text_neg
                if self.tokenizer is not None:
                    data["text"] = self.tokenizer(
                        data["text"], return_tensors="pt", padding=True, truncation=True
                    )
                data["text"] = {
                    key: val.to(self.device) for key, val in data["text"].items()
                }
                data["video"] = data["video"].to(self.device)
                if self.config["loss"]["type"] == "EgoNCE_hal":
                    n_embeds = data["noun_vec"].to(self.device)
                    v_embeds = data["verb_vec"].to(self.device)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    text_embeds, video_embeds = self.model(data)
                    if "text_neg" not in data.keys():
                        video_embeds = self.allgather(
                            video_embeds, self.n_gpu, self.args
                        )
                        text_embeds = self.allgather(text_embeds, self.n_gpu, self.args)
                        output = sim_matrix(text_embeds, video_embeds)
                        # output_hal = sim_matrix(text_embeds{:}, video_embeds)
                        # if self.config['loss']['type'] == 'EgoNCE':
                        #     sim_v = sim_matrix(v_embeds, v_embeds)
                        #     sim_n = sim_matrix(n_embeds, n_embeds)
                        #     loss = self.loss(output, sim_v, sim_n)
                        # else:
                        loss = self.loss(output)
                    else:
                        # text_embeds = self.allgather(text_embeds, self.n_gpu, self.args)
                        pos_text = text_embeds[: video_embeds.shape[0]]
                        pos_text = self.allgather(pos_text, self.n_gpu, self.args)
                        neg_text = text_embeds[video_embeds.shape[0] :]
                        neg_text = self.allgather(neg_text, self.n_gpu, self.args)
                        video_embeds = self.allgather(
                            video_embeds, self.n_gpu, self.args
                        )
                        # neg_num = self.config['data_loader'][0]['args']['neg_num']
                        output = sim_matrix(pos_text, video_embeds)
                        output_hal = sim_matrix(neg_text, video_embeds)
                        if self.config["loss"]["type"] == "EgoNCE_hal":
                            n_embeds = self.allgather(n_embeds, self.n_gpu, self.args)
                            v_embeds = self.allgather(v_embeds, self.n_gpu, self.args)
                            sim_v = sim_matrix(v_embeds, v_embeds)
                            sim_n = sim_matrix(n_embeds, n_embeds)
                            loss = self.loss(output, output_hal, sim_v, sim_n)
                        else:
                            loss = self.loss(output, output_hal)

                loss.backward()

                self.optimizer.step()
                self.lr_scheduler.step()
                if self.writer is not None and self.args.rank == 0:
                    # self.writer.log_scalar(f'loss_train_{dl_idx}', loss.detach().item())
                    total = int(self.data_loader[dl_idx].n_samples / self.n_gpu)
                    current = batch_idx * self.data_loader[dl_idx].batch_size
                    final_total = (epoch - 1) * total + current
                    self.writer.add_scalar(
                        f"Loss_training/loss_{dl_idx}",
                        loss.detach().item(),
                        final_total,
                    )

                total_loss[dl_idx] += loss.detach().item()

                # if batch_idx % self.log_step == 0 and self.args.local_rank == 0:
                if batch_idx % self.log_step == 0 and self.args.rank == 0:
                    current_lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
                    self.logger.info(
                        "Train Epoch: {} dl{} {} Loss: {:.6f}; lr: {}; Time/iteration: {:.3f}m; Time so far/epoch: {:.3f}h".format(
                            epoch,
                            dl_idx,
                            self._progress(batch_idx, dl_idx),
                            loss.detach().item(),
                            current_lr,
                            (time.time() - since) / 60,
                            (time.time() - start) / 60 / 60,
                        )
                    )
                    since = time.time()

                self.optimizer.zero_grad()
            if batch_idx == self.len_epoch:
                break

        log = {
            f"loss_{dl_idx}": total_loss[dl_idx] / self.len_epoch
            for dl_idx in range(len(self.data_loader))
        }

        if self.writer is not None and self.args.rank == 0:
            for dl_idx in range(len(self.data_loader)):
                tl = total_loss[dl_idx] / self.len_epoch
                self.writer.add_scalar(
                    f"Loss_training/loss_total_{dl_idx}", tl, epoch - 1
                )

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            if self.args.rank == 0:
                log.update(val_log)

        # self._adjust_learning_rate(self.optimizer, epoch, self.args)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        # total_val_metrics = [np.zeros(len(self.metrics))] * len(self.valid_data_loader)

        gt_arr = {x: [] for x in range(len(self.valid_data_loader))}
        pred_arr = {x: [] for x in range(len(self.valid_data_loader))}
        type_arr = {x: [] for x in range(len(self.valid_data_loader))}

        with torch.no_grad():
            # for validation we switch the nested loop order, because alternate batches not needed...
            # ... and dataloaders can be of different length
            for dl_idx, dl in enumerate(self.valid_data_loader):
                for batch_idx, data in enumerate(tqdm(dl)):
                    # v2t
                    # remove batch (batch_size == 1)
                    data["video"] = data["video"][0][data["correct"][0]].unsqueeze(0)
                    data["text"] = []
                    for text_tmp in data["text_ops"]:
                        data["text"] += list(text_tmp)
                    ###### debug ######
                    # raw_text = data["text"]
                    ###### debug ######
                    if self.tokenizer is not None:
                        data["text"] = self.tokenizer(
                            data["text"],
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                        )

                    data["text"] = {
                        key: val.to(self.device) for key, val in data["text"].items()
                    }
                    data["video"] = data["video"].to(self.device)
                    text_embed, vid_embed = self.model(data, return_embeds=True)

                    data_gt = data["correct"][0].to(self.device).unsqueeze(0)
                    data_pred = sim_matrix(vid_embed, text_embed)
                    data_type = data["type"][0].to(self.device).unsqueeze(0)

                    ## t2v
                    # data['video'] = data['video'][0]  # remove batch
                    # data['text'] = data['text']

                    # if self.tokenizer is not None:
                    #     data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                    # data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
                    # data['video'] = data['video'].to(self.device)
                    # text_embed, vid_embed = self.model(data, return_embeds=True)

                    # data_gt = data['correct'][0].to(self.device).unsqueeze(0)
                    # data_pred = sim_matrix(text_embed, vid_embed)
                    # data_type = data['type'][0].to(self.device).unsqueeze(0)

                    # if isinstance(self.model, nn.DataParallel) and data["video"].shape[0] < len(self.model.device_ids):
                    # # Note that if some batch has size smaller than the GPU size, `DataParallel` will fail.
                    # # It can happen with the last batch of the dataset, depending on its size.
                    # # This avoids using `DataParallel` in this case, and supposes the entire batch fits in one GPU.
                    #    text_embed, vid_embed = self.model.module(data, return_embeds=True)
                    # else:
                    #    text_embed, vid_embed = self.model(data, return_embeds=True)

                    data_gt_all = [torch.zeros_like(data_gt) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(data_gt_all, data_gt)
                    data_gt_all = torch.cat(data_gt_all, dim=0)

                    data_pred_all = [
                        torch.zeros_like(data_pred) for _ in range(self.n_gpu)
                    ]
                    torch.distributed.all_gather(data_pred_all, data_pred)
                    data_pred_all = torch.cat(data_pred_all, dim=0)

                    data_type_all = [
                        torch.zeros_like(data_type) for _ in range(self.n_gpu)
                    ]
                    torch.distributed.all_gather(data_type_all, data_type)
                    data_type_all = torch.cat(data_type_all, dim=0)

                    gt_arr[dl_idx].append(data_gt_all.cpu())
                    pred_arr[dl_idx].append(data_pred_all.cpu())
                    type_arr[dl_idx].append(data_type_all.cpu())

            if self.writer is not None and self.args.rank == 0:
                for dl_idx in range(len(self.valid_data_loader)):
                    tl = total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                    self.writer.add_scalar(
                        f"Loss_val/loss_total_{dl_idx}", tl, epoch - 1
                    )

        for dl_idx in range(len(self.valid_data_loader)):
            nested_metrics = {x: {} for x in range(len(self.valid_data_loader))}

            gt_arr_cat = torch.cat(gt_arr[dl_idx])
            pred_arr_cat = torch.cat(pred_arr[dl_idx])
            type_cat = torch.cat(type_arr[dl_idx])

            for metric in self.metrics:
                metric_name = metric.__name__
                res = metric(pred_arr_cat, gt_arr_cat, type_cat)
                if self.args.rank == 0:
                    self.logger.info(
                        verbose(
                            epoch=epoch,
                            metrics=res,
                            name=self.valid_data_loader[dl_idx].dataset_name,
                        )
                    )
                nested_metrics[dl_idx][metric_name] = res
                if self.writer is not None and self.args.rank == 0:
                    to_write = format_nested_metrics_for_writer(
                        res,
                        mode=metric_name,
                        name=self.valid_data_loader[dl_idx].dataset_name,
                    )
                    # for key, val in to_write.items():
                    #     self.writer.log_scalar(key, val)
                    for key, val in to_write.items():
                        key = key.replace("[", "_").replace("]", "_")
                        self.writer.add_scalar(
                            f"Val_metrics_{dl_idx}/{key}", val, epoch - 1
                        )

        res_dict = {}
        if self.args.rank == 0:
            res_dict = {
                f"val_loss_{dl_idx}": total_val_loss[dl_idx]
                / len(self.valid_data_loader[dl_idx])
                for dl_idx in range(len(self.valid_data_loader))
            }
            res_dict["nested_val_metrics"] = nested_metrics

        return res_dict

    def _progress(self, batch_idx, dl_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader[dl_idx], "n_samples"):
            current = batch_idx * self.data_loader[dl_idx].batch_size
            total = int(self.data_loader[dl_idx].n_samples / self.n_gpu)
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


def verbose(epoch, metrics, name="TEST"):
    msg = ""
    for key in metrics.keys():
        acc = metrics[key]
        msg += f"{name:s} epoch {epoch}, {key:s}, Acc: {acc:.1f};    "
    print(msg)
    return msg


def format_nested_metrics_for_writer(metrics, mode, name="TEST"):
    res = {}
    for key, val in metrics.items():
        log_name = f"[{mode}]{name}_{key}"
        res[log_name] = val
    return res
