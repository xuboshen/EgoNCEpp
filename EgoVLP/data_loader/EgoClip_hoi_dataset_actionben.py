import os
import sys
import json
import pandas as pd
import random

from base.base_dataset import TextVideoDataset
from data_loader.EgoClip_EgoMCQ_dataset import EgoClip_EgoMCQ
from data_loader.transforms import init_transform_dict, init_video_transform_dict

import torch
from PIL import Image
from torchvision import transforms

class EgoClip_HOI(EgoClip_EgoMCQ):
    def _load_metadata(self):
        split_files = {
            'train': str(self.trainset_name),
            'val': 'actionbench_test.csv',
            'test': 'actionbench_test.csv'
        }
        target_split_fp = split_files[self.split]
        if str(self.trainset_name) == 'egomcq_bad_case.json':
            self.split = 'val'
            target_split_fp = str(self.trainset_name)
        self.chunk_sec = 300  # Each segment is up to 600s
        self.noun_dim = 582  # num of nouns of ego4d taxonomy dictionary
        self.verb_dim = 118  # num of verbs of ego4d taxonomy dictionary

        if self.split == 'train':
            self.metadata = pd.read_csv(os.path.join(self.meta_dir, target_split_fp), sep='\t', on_bad_lines='skip')
            self.frame_sample = 'rand'

            # if self.neg_param:
            #     self.metadata['chunk_id'] = self.metadata['narration_time'] // self.neg_param
            #     self.metadata['chunk_id'] = self.metadata['chunk_id'].astype(str)
            #     self.metadata['segment_id'] = self.metadata['video_uid'] + '_' + self.metadata['chunk_id']

        elif self.split in ['val', 'test']:
            self.frame_sample = 'uniform'
            self.metadata = pd.read_csv(os.path.join(self.meta_dir, target_split_fp), sep='\t', on_bad_lines='skip')

    def _get_video_path(self, sample):
        video_uid = sample['video_uid']
        video_start_sec = max(float(sample['clip_start']), 0)
        video_end_sec   = max(float(sample['clip_end']), 0)

        video_fp = [self.data_dir, video_uid]

        start_sec, end_sec = video_start_sec, video_end_sec
        return video_fp, start_sec, end_sec

    def _get_video_frames(self, video_fp, start_sec, end_sec, clip_length, jitter=False):
        video_loading = self.video_params.get('loading', 'strict')
        try:
            if os.path.exists(os.path.join(video_fp[0], video_fp[1] + '.mp4')):
                imgs = self.video_reader(video_fp[0], video_fp[1], start_sec,
                        end_second=end_sec,
                        clip_length=clip_length,
                        jitter=jitter)
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        except Exception as e:
            if video_loading == 'strict':
                raise ValueError(
                    f'Video loading failed for {video_fp}, video loading for this dataset is strict.') from e
            else:
                imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
                imgs = transforms.ToTensor()(imgs).unsqueeze(0).repeat(clip_length, 1, 1, 1).transpose(1, 3)

        if self.transforms is not None:
            if self.video_params['num_frames'] > 1:
                imgs = imgs.permute(3, 0, 1, 2)  # [T, H, W, C] ---> [C, T, H, W]
                imgs = self.transforms(imgs)
                imgs = imgs.transpose(0, 1)  # recover
            else:
                imgs = self.transforms(imgs.squeeze(0).permute(2, 0, 1))

        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs
        return final

    def _get_caption(self, sample):
        noun_vec = torch.zeros(self.noun_dim)
        verb_vec = torch.zeros(self.verb_dim)
        noun_idx = eval(sample['tag_noun'])
        verb_idx = eval(sample['tag_verb'])
        for i in noun_idx:
            noun_vec[i] = 1
        for i in verb_idx:
            verb_vec[i] = 1

        return sample['clip_text'], noun_vec, verb_vec


    def _get_train_item(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, start_sec, end_sec = self._get_video_path(sample)
        caption, noun_vec, verb_vec = self._get_caption(sample)
        final = self._get_video_frames(video_fp, start_sec, end_sec, clip_length=self.video_params['num_frames'], jitter=True)
        
        # Scene-aware negative sampling
        if self.neg_param:
            if self.neg_param == 1:
                try:
                    verb_caption_neg = self._get_caption_neg(sample, "verb_neg")
                except:
                    sample_temp = self.metadata.iloc[item + 1]
                    verb_caption_neg = self._get_caption_neg(sample_temp, "verb_neg")
                caption_neg = verb_caption_neg
            elif self.neg_param == 2:
                try:
                    noun_caption_neg = self._get_caption_neg(sample, "noun_neg")
                except:
                    sample_temp = self.metadata.iloc[item + 1]
                    noun_caption_neg = self._get_caption_neg(sample_temp, "noun_neg")
                caption_neg = noun_caption_neg
            elif self.neg_param in [3, 4, 5, 6]:
                try:
                    verb_caption_neg = self._get_caption_neg(sample, "verb_neg")
                    noun_caption_neg = self._get_caption_neg(sample, "noun_neg")
                except:
                    sample_temp = self.metadata.iloc[item + 1]
                    verb_caption_neg = self._get_caption_neg(sample_temp, "verb_neg")
                    noun_caption_neg = self._get_caption_neg(sample_temp, "noun_neg")
                caption_neg = ','.join(verb_caption_neg.split(',') + noun_caption_neg.split(','))

        meta_arr = {'raw_captions': caption, 'paths': video_fp, 'dataset': self.dataset_name}
        if self.neg_param:
            if self.video_params['egonce'] == True or self.neg_param in [4, 5, 6]:
                return {'video': final, 'text': caption, 'text_neg': caption_neg,
                    'noun_vec': noun_vec, 'verb_vec': verb_vec,
                    'meta': meta_arr}
            else:
                return {'video': final, 'text': caption, 'text_neg': caption_neg,
                    'meta': meta_arr}
        else:
            return {'video': final, 'text': caption,
                'meta': meta_arr}

    def _get_caption_neg(self, sample, key):
        caption_neg = str(sample[key]).split(',')
        if len(caption_neg) < self.neg_num:
            for i in range(len(caption_neg), self.neg_num):
                caption_neg.append(caption_neg[-1])
        elif len(caption_neg) > self.neg_num:
            caption_neg = caption_neg[:self.neg_num]
        caption_neg = ','.join(caption_neg)
        return caption_neg
    
    def _get_caption_neg_val(self, sample, key):
        caption_neg = str(sample[key]).split(',')
        if len(caption_neg) < 10:
            for i in range(len(caption_neg), 10):
                caption_neg.append(caption_neg[-1])
        elif len(caption_neg) > 10:
            caption_neg = caption_neg[:10]
        caption_neg = ','.join(caption_neg)
        return caption_neg
    
    def _get_val_item(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, start_sec, end_sec = self._get_caption(sample)
        caption, noun_vec, verb_vec = self._get_caption(sample)
        final = self._get_video_frames(video_fp, start_sec, end_sec, clip_length=self.video_params['num_frames'], jitter=True)

        try:
            verb_caption_neg = self._get_caption_neg_val(sample, "action_antonym_clip_text")
            noun_caption_neg = self._get_caption_neg_val(sample, "action_antonym_clip_text")
        except:
            sample_temp = self.metadata.iloc[item + 1]
            verb_caption_neg = self._get_caption_neg_val(sample_temp, "action_antonym_clip_text")
            noun_caption_neg = self._get_caption_neg_val(sample_temp, "action_antonym_clip_text")

        meta_arr = {'raw_captions': caption, 'paths': video_fp, 'dataset': self.dataset_name}
        return {'video': final, 'text': caption, 'verb_neg': verb_caption_neg, 'noun_neg': noun_caption_neg,
            'meta': meta_arr}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        try:
            if self.split == 'train':
                return self._get_train_item(item)
            elif self.split in ['val', 'test']:
                return self._get_val_item(item)
        except Exception as e:
            raise ValueError(e)
            assert False
            print(e)
            return self.__getitem__(random.randint(0, self.__len__() - 1))
            
