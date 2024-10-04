import os

import torch
import torch.nn.functional as F
import yaml
from model.model import FrozenInTime
from peft import LoraConfig, get_peft_model
from utils.util import state_dict_data_parallel_fix

with open("./EgoNCE_MLM_ITM_Config.yml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


class EgoVLP_lora(FrozenInTime):
    def __init__(
        self,
        video_params,
        text_params,
        projection_dim=4096,
        load_checkpoint=None,
        projection="minimal",
        load_temporal_fix="bilinear",
        config=config,
        task_names="EgoNCE_ITM_MLM",
        norm_layer=None,
        embed_dim=768,
        lora_params=None,
    ):
        print("-" * 100, load_checkpoint)
        super().__init__(
            video_params,
            text_params,
            projection_dim,
            load_checkpoint,
            projection,
            load_temporal_fix,
            config,
            task_names,
            norm_layer,
            embed_dim,
        )
        self.lora_r = lora_params["lora_r"]
        self.lora_alpha = lora_params["lora_alpha"]
        self.lora_dropout = lora_params["lora_dropout"]
        self.add_time_attn = True
        self.convert_to_lora()
        self.freeze_backbone()
        if "lora" in load_checkpoint:
            self.load_checkpoints(load_checkpoint)

    def load_checkpoints(self, load_checkpoint):
        print("loading lora checkpoints...")
        # checkpoint = torch.load(load_checkpoint)
        try:
            local_rank = int(os.environ["LOCAL_RANK"])  # fixed by qinghong.
        except Exception as E:
            print(E)
            print("False Local rank, set to 0 instead")
            local_rank = 0
        checkpoint = torch.load(
            load_checkpoint, map_location="cuda:{}".format(local_rank)
        )
        state_dict = checkpoint["state_dict"]
        new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
        new_state_dict = self._inflate_positional_embeds_lora(new_state_dict)
        self.load_state_dict(new_state_dict, strict=True)

    def convert_to_lora(self):
        if self.lora_r == 0:
            return
        target_modules = ["attn.qkv", "attn.proj"]  # why not finetune MLP for images?
        if self.add_time_attn:
            target_modules.extend(
                ["timeattn.qkv", "timeattn.proj", "mlp.fc1", "mlp.fc2"]
            )
        config = LoraConfig(
            r=self.lora_r,  # 16
            lora_alpha=self.lora_alpha,  # 16
            target_modules=target_modules,  # self_attn.out_proj
            lora_dropout=self.lora_dropout,  # 0.1
            bias="none",
            modules_to_save=[],
        )
        self.video_model = get_peft_model(self.video_model, config)

    def freeze_backbone(self):
        for param in self.text_model.parameters():
            param.requires_grad = False
        for param in self.txt_proj.parameters():
            param.requires_grad = False

        for name, param in self.video_model.named_parameters():
            if "lora" not in name.lower() and "norm" not in name.lower():
                param.requires_grad = False
            else:
                param.requires_grad = True
        for name, param in self.vid_proj.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False
            else:
                param.requires_grad = True
        self.video_model.cls_token.requires_grad = True

    def _inflate_positional_embeds_lora(self, new_state_dict):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if (
            "video_model.base_model.model.temporal_embed" in new_state_dict
            and "video_model.base_model.model.temporal_embed" in curr_keys
        ):
            load_temporal_embed = new_state_dict[
                "video_model.base_model.model.temporal_embed"
            ]
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.video_params["num_frames"]
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    print(
                        f'### loaded {self.video_params["model"]} model has MORE frames than current...'
                        f"### loading weights, filling in the extras via {self.load_temporal_fix}"
                    )
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    print(
                        f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
                        f"### loading weights, filling in the extras via {self.load_temporal_fix}"
                    )
                    if self.load_temporal_fix == "zeros":
                        new_temporal_embed = torch.zeros(
                            [load_temporal_embed.shape[0], curr_num_frames, embed_dim]
                        )
                        new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    elif self.load_temporal_fix in ["interp", "bilinear"]:
                        # interpolate
                        # unsqueeze so pytorch thinks its an image
                        mode = "nearest"
                        if self.load_temporal_fix == "bilinear":
                            mode = "bilinear"
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(
                            load_temporal_embed,
                            (curr_num_frames, embed_dim),
                            mode=mode,
                            align_corners=True,
                        ).squeeze(0)
                    else:
                        raise NotImplementedError
                new_state_dict[
                    "video_model.base_model.model.temporal_embed"
                ] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if (
            "video_model.pos_embed" in new_state_dict
            and "video_model.pos_embed" in curr_keys
        ):
            load_pos_embed = new_state_dict["video_model.pos_embed"]
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = self.state_dict()["video_model.pos_embed"]
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    "Loading models with different spatial resolution / patch number not yet implemented, sorry."
                )

        return new_state_dict
