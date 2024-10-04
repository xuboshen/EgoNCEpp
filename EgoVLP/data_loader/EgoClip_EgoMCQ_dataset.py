import json
import os
import random
import sys

import pandas as pd
import torch
from base.base_dataset import TextVideoDataset
from data_loader.transforms import init_transform_dict, init_video_transform_dict
from PIL import Image
from torchvision import transforms


class EgoClip_EgoMCQ(TextVideoDataset):
    def _load_metadata(self):
        split_files = {
            "train": str(self.trainset_name),
            "val": "egohoi.json",
            "test": "egohoi.json",
        }
        target_split_fp = split_files[self.split]

        self.chunk_sec = 300  # Each segment is up to 600s
        self.noun_dim = 582  # num of nouns of ego4d taxonomy dictionary
        self.verb_dim = 118  # num of verbs of ego4d taxonomy dictionary

        if self.split == "train":
            self.metadata = pd.read_csv(
                os.path.join(self.meta_dir, target_split_fp),
                sep="\t",
                on_bad_lines="skip",
            )
            self.frame_sample = "rand"

            if self.neg_param:
                self.metadata["chunk_id"] = (
                    self.metadata["narration_time"] // self.neg_param
                )
                self.metadata["chunk_id"] = self.metadata["chunk_id"].astype(str)
                self.metadata["segment_id"] = (
                    self.metadata["video_uid"] + "_" + self.metadata["chunk_id"]
                )

        elif self.split in ["val", "test"]:
            self.frame_sample = "uniform"
            with open(os.path.join(self.meta_dir, target_split_fp), "r") as load_f:
                self.metadata = json.load(load_f)

    def _get_video_path(self, sample):
        video_uid = sample["video_uid"]
        video_start_sec = max(float(sample["clip_start"]), 0)
        video_end_sec = max(float(sample["clip_end"]), 0)

        video_fp = [self.data_dir, video_uid]

        start_sec, end_sec = video_start_sec, video_end_sec
        return video_fp, start_sec, end_sec

    def _get_video_frames(
        self, video_fp, start_sec, end_sec, clip_length, jitter=False
    ):
        video_loading = self.video_params.get("loading", "strict")
        try:
            if os.path.exists(os.path.join(video_fp[0], video_fp[1] + ".mp4")):
                imgs = self.video_reader(
                    video_fp[0],
                    video_fp[1],
                    start_sec,
                    end_second=end_sec,
                    clip_length=clip_length,
                    jitter=jitter,
                )
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        except Exception as e:
            if video_loading == "strict":
                raise ValueError(
                    f"Video loading failed for {video_fp}, video loading for this dataset is strict."
                ) from e
            else:
                imgs = Image.new(
                    "RGB",
                    (self.video_params["input_res"], self.video_params["input_res"]),
                    (0, 0, 0),
                )
                imgs = (
                    transforms.ToTensor()(imgs)
                    .unsqueeze(0)
                    .repeat(clip_length, 1, 1, 1)
                    .transpose(1, 3)
                )

        if self.transforms is not None:
            if self.video_params["num_frames"] > 1:
                imgs = imgs.permute(3, 0, 1, 2)  # [T, H, W, C] ---> [C, T, H, W]
                imgs = self.transforms(imgs)
                imgs = imgs.transpose(0, 1)  # recover
            else:
                imgs = self.transforms(imgs.squeeze(0).permute(2, 0, 1))

        final = torch.zeros(
            [
                self.video_params["num_frames"],
                3,
                self.video_params["input_res"],
                self.video_params["input_res"],
            ]
        )
        final[: imgs.shape[0]] = imgs
        return final

    def _get_caption(self, sample):
        noun_vec = torch.zeros(self.noun_dim)
        verb_vec = torch.zeros(self.verb_dim)
        noun_idx = eval(sample["tag_noun"])
        verb_idx = eval(sample["tag_verb"])
        for i in noun_idx:
            noun_vec[i] = 1
        for i in verb_idx:
            verb_vec[i] = 1

        return sample["clip_text"], noun_vec, verb_vec

    def _get_train_item(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, start_sec, end_sec = self._get_video_path(sample)
        caption, noun_vec, verb_vec = self._get_caption(sample)
        final = self._get_video_frames(
            video_fp,
            start_sec,
            end_sec,
            clip_length=self.video_params["num_frames"],
            jitter=True,
        )

        # Scene-aware negative sampling
        if self.neg_param:
            # sample_neg = self.metadata[(self.metadata.video_uid==sample.video_uid)].sample(1).iloc[0] # variant of negative sample from same video
            sample_neg = (
                self.metadata[self.metadata.segment_id == sample.segment_id]
                .sample(1)
                .iloc[0]
            )
            video_fp_neg, start_sec_neg, end_sec_neg = self._get_video_path(sample_neg)
            caption_neg, noun_vec_neg, verb_vec_neg = self._get_caption(sample_neg)
            final_neg = self._get_video_frames(
                video_fp_neg,
                start_sec_neg,
                end_sec_neg,
                clip_length=self.video_params["num_frames"],
                jitter=True,
            )

        meta_arr = {
            "raw_captions": caption,
            "paths": video_fp,
            "dataset": self.dataset_name,
        }
        if self.neg_param:
            return {
                "video": final,
                "text": caption,
                "video_neg": final_neg,
                "text_neg": caption_neg,
                "meta": meta_arr,
                "noun_vec": noun_vec,
                "verb_vec": verb_vec,
                "noun_vec_neg": noun_vec_neg,
                "verb_vec_neg": verb_vec_neg,
            }
        else:
            return {
                "video": final,
                "text": caption,
                "meta": meta_arr,
                "noun_vec": noun_vec,
                "verb_vec": verb_vec,
            }

    def _get_val_item(self, item):
        self.video_params["num_frames"] = 4
        item = item % len(self.metadata)
        itemMCQ = self.metadata[list(self.metadata.keys())[item]]

        answerIndex = itemMCQ["answer"]
        sampleQuery = itemMCQ["query"]
        textQuery, _, _ = self._get_caption(sampleQuery)
        meta_arr = {"meta": itemMCQ}

        sampleOptions = itemMCQ["choices"]
        num_options = len(sampleOptions)
        textOptions = []
        videoOptions = torch.zeros(
            [
                num_options,
                self.video_params["num_frames"],
                3,
                self.video_params["input_res"],
                self.video_params["input_res"],
            ]
        )

        for id, option in enumerate(sampleOptions):
            sampleOptioni = sampleOptions[option]
            video_fp, start_sec, end_sec = self._get_video_path(sampleOptioni)
            caption, _, _ = self._get_caption(sampleOptioni)
            textOptions.append(caption)
            imgs = self._get_video_frames(
                video_fp,
                start_sec,
                end_sec,
                clip_length=self.video_params["num_frames"],
                jitter=False,
            )
            videoOptions[id] = imgs

        type = itemMCQ["types"]  # 1 for inter; 2 for intra
        data = {
            "video": videoOptions,
            "text": textQuery,
            "text_ops": textOptions,
            "correct": answerIndex,
            "type": type,
            "meta": meta_arr,
        }
        return data

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        try:
            if self.split == "train":
                return self._get_train_item(item)
            elif self.split in ["val", "test"]:
                return self._get_val_item(item)
        except Exception as e:
            print(f"Error with {e}")
            return self.__getitem__(random.randint(0, self.__len__() - 1))
