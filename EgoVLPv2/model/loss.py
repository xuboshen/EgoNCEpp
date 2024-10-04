# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pdb
import pickle

import torch
import torch.nn.functional as F
from torch import nn


class NormSoftmaxLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()

        self.temperature = temperature

    def forward(self, x):
        "Assumes input x is similarity matrix of N x M in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_logsm = F.log_softmax(x / self.temperature, dim=1)
        j_logsm = F.log_softmax(x.t() / self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return -loss_i - loss_j, self.temperature


class EgoNCE(nn.Module):
    def __init__(self, temperature=0.05, noun=True, verb=True):
        super().__init__()
        self.noun = noun
        self.verb = verb
        self.temperature = temperature

    def forward(self, x, mask_v, mask_n):
        mask_diag = torch.eye(x.shape[0]).cuda()
        if self.noun and self.verb:
            mask = mask_v * mask_n + mask_diag
        elif self.noun and not self.verb:
            mask = mask_n + mask_diag
        elif self.verb and not self.noun:
            mask = mask_v + mask_diag
        else:
            mask = mask_diag

        "Assumes input x is similarity matrix of N x M in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_sm = F.softmax(x / self.temperature, dim=1)
        j_sm = F.softmax(x.t() / self.temperature, dim=1)

        mask_bool = mask > 0
        idiag = torch.log(torch.sum(i_sm * mask_bool, dim=1))
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.log(torch.sum(j_sm * mask_bool, dim=1))
        loss_j = jdiag.sum() / len(jdiag)
        return -loss_i - loss_j, mask_bool, self.temperature
        # return - loss_i - loss_j


class Hal_SoftmaxLoss_t2v(nn.Module):
    def __init__(self, temperature=0.05, noun=True):
        super().__init__()
        self.noun = noun
        self.temperature = temperature

    def forward(self, x, hal_x, mask_n):
        "Assumes input x is similarity matrix of N x N in [-1, 1], computed using the cosine similarity between normalised vectors"
        "        input hal_x is                  10*N x N in [-1, 1]"
        mask_diag = torch.eye(x.shape[0]).cuda()
        if self.noun:
            mask = mask_n + mask_diag
        i_sm = F.softmax(x / self.temperature, dim=1)
        # j_logsm = F.log_softmax(x.t()/self.temperature, dim=1)
        # sum over positives
        mask_bool = mask > 0
        i_mask = torch.zeros(i_sm.shape).to(mask_bool.device) + 1e-6
        i_mask[: mask_bool.shape[0], : mask_bool.shape[1]] = mask_bool
        idiag = torch.log(torch.sum(i_sm * i_mask, dim=1))
        loss_i = idiag.sum() / len(idiag)
        # jdiag = torch.diag(j_logsm)
        # loss_j = jdiag.sum() / len(jdiag)
        neg_num, video_num = hal_x.shape[0] // hal_x.shape[1], hal_x.shape[1]
        # print(hal_x.shape)
        hal_x = torch.stack(
            [hal_x[i * neg_num : (i + 1) * neg_num, i] for i in range(video_num)], dim=1
        )  # neg, video_num

        # hal_x= hal_x.reshape(video_num, neg_num, video_num)[:,:,0].reshape(10, -1)
        # hal_x = torch.diagonal(hal_x.reshape(neg_num, video_num, video_num), dim1=1, dim2=2)
        x = torch.cat([x, hal_x], dim=0)
        # sum over positives
        # import pdb; pdb.set_trace()
        j_logsm = F.log_softmax(x.t() / self.temperature, dim=1)
        jdiag = torch.diag(j_logsm)
        loss_hal = jdiag.sum() / len(jdiag)

        original_cont = -loss_i  # - loss_j
        loss_hal = -loss_hal
        return original_cont + loss_hal


class Hal_SoftmaxLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()

        self.temperature = temperature

    def forward(self, x, hal_x):
        "Assumes input x is similarity matrix of N x N in [-1, 1], computed using the cosine similarity between normalised vectors"
        "        input hal_x is                  10*N x N in [-1, 1]"
        # infonce
        i_logsm = F.log_softmax(x / self.temperature, dim=1)
        # j_logsm = F.log_softmax(x.t()/self.temperature, dim=1)
        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)
        # jdiag = torch.diag(j_logsm)
        # loss_j = jdiag.sum() / len(jdiag)

        neg_num, video_num = hal_x.shape[0] // hal_x.shape[1], hal_x.shape[1]
        # print(hal_x.shape)
        hal_x = torch.stack(
            [hal_x[i * neg_num : (i + 1) * neg_num, i] for i in range(video_num)], dim=1
        )  # neg, video_num

        # hal_x= hal_x.reshape(video_num, neg_num, video_num)[:,:,0].reshape(10, -1)
        # hal_x = torch.diagonal(hal_x.reshape(neg_num, video_num, video_num), dim1=1, dim2=2)
        x = torch.cat([x, hal_x], dim=0)
        # sum over positives
        # import pdb; pdb.set_trace()
        j_logsm = F.log_softmax(x.t() / self.temperature, dim=1)
        jdiag = torch.diag(j_logsm)
        loss_hal = jdiag.sum() / len(jdiag)

        original_cont = -loss_i  # - loss_j
        loss_hal = -loss_hal
        return original_cont + loss_hal


class MaxMarginRankingLoss(nn.Module):
    def __init__(self, margin=0.2, fix_norm=True):
        super().__init__()
        self.fix_norm = fix_norm
        self.loss = nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, x, weight=None):
        n = x.size()[0]

        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = torch.cat((x2, x3), 0)
        max_margin = F.relu(self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = torch.ones(x.shape) - torch.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = torch.index_select(x1, dim=0, index=keep_idx)
            x2_ = torch.index_select(x2, dim=0, index=keep_idx)
            max_margin = F.relu(self.margin - (x1_ - x2_))

        return max_margin.mean()


class AdaptiveMaxMarginRankingLoss(nn.Module):
    def __init__(self, margin=0.4, fix_norm=True):
        super().__init__()
        self.fix_norm = fix_norm
        self.loss = nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, x, weight=None):
        n = x.size()[0]

        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)

        w1 = weight.unsqueeze(1)
        w1 = w1.expand(n, n)
        w1 = w1.contiguous().view(-1, 1)
        w1 = torch.cat((w1, w1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = torch.cat((x2, x3), 0)
        max_margin = F.relu(w1 * self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = torch.ones(x.shape) - torch.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = torch.index_select(x1, dim=0, index=keep_idx)
            w1_ = torch.index_select(w1, dim=0, index=keep_idx)
            x2_ = torch.index_select(x2, dim=0, index=keep_idx)
            max_margin = F.relu(w1_ * self.margin - (x1_ - x2_))

        return max_margin.mean()


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.loss(output, target)
