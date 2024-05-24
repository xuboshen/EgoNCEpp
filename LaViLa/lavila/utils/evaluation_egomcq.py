# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def egomcq_accuracy_metrics(preds, labels, types):
    metrics = {}
    type_list = torch.unique(types)
    group_list = ["Intra-video", "Inter-video"]
    for type_i, group_i in zip(type_list, group_list):
        correct = 0
        total = 0
        for pred, label, type in zip(preds, labels, types):
            if type == type_i:
                pred_ = torch.argmax(pred)
                if pred_.item() == label.item():
                    correct += 1
                total += 1
        accuracy = correct/total
        metrics[group_i] = accuracy * 100
    return metrics


def egomcqv2t_accuracy_metrics(preds, labels, types):
    metrics = {}
    type_list = torch.unique(types)
    group_list = ["Intra-video", "Inter-video"]
    for type_i, group_i in zip(type_list, group_list):
        correct = 0
        total = 0
        for pred, label, type in zip(preds, labels, types):
            if type == type_i:
                pred_ = torch.argmax(pred)
                if pred_.item() == label.item():
                    correct += 1
                total += 1
        accuracy = correct/total
        metrics[group_i] = accuracy * 100
    return metrics

def egohoi_accuracy_metrics(preds):
    metrics = {}
    group_list = ["Verb", "Noun", "Verb and noun"]
    import pdb;
    # verb
    correct = 0
    total = 0
    for pred in preds:
        neg_num = (len(pred) - 1) // 2
        res = pred[:neg_num + 1]
        label = torch.argmax(res).item()
        if label == 0:
            correct += 1
        total += 1
    acc = correct * 1.0 / total
    metrics[group_list[0]] = acc * 100


    # noun 
    correct = 0
    total = 0
    for pred in preds:
        neg_num = (len(pred) - 1) // 2
        res = torch.cat((pred[0].unsqueeze(0), pred[neg_num + 1:]))
        label = torch.argmax(res).item()
        if label == 0:
            correct += 1
        total += 1
    acc = correct * 1.0 / total
    metrics[group_list[1]] = acc * 100


    # verb and noun  
    correct = 0
    total = 0
    for pred in preds:
        neg_num = (len(pred) - 1) // 2
        res = pred
        label = torch.argmax(res).item()
        if label == 0:
            correct += 1
        total += 1
    acc = correct * 1.0 / total
    metrics[group_list[2]] = acc * 100


    # for type_i, group_i in zip(type_list, group_list):
    #     correct = 0
    #     total = 0
    #     for pred, label, type in zip(preds, labels, types):
    #         if type == type_i:
    #             pred_ = torch.argmax(pred)
    #             if pred_.item() == label.item():
    #                 correct += 1
    #             total += 1
    #     accuracy = correct/total
    #     metrics[group_i] = accuracy * 100
    return metrics
