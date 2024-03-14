import os
import pdb

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from base import BaseModel
from model.audio_encoder import AudioEncoder
from model.loss import AudioTextContrastiveLoss, NTXent
from model.text_encoder import TextEncoder
from model.video_transformer import SpaceTimeTransformer
from utils.util import state_dict_data_parallel_fix


class FrozenInTime(BaseModel):
    def __init__(
        self,
        video_params,
        text_params,
        projection_dim=256,
        load_checkpoint=None,
        projection="minimal",
        load_temporal_fix="zeros",
    ):
        super().__init__()

        self.video_params = video_params
        self.text_params = text_params
        self.load_temporal_fix = load_temporal_fix
        if not text_params["pretrained"]:
            raise NotImplementedError(
                "Huggingface text models require pretrained init."
            )

        # pdb.set_trace()
        if self.text_params["model"].startswith("distilbert"):
            self.text_model = AutoModel.from_pretrained(
                "distilbert-base-uncased",
                cache_dir="pretrained/distilbert-base-uncased",
            )
        else:
            self.text_model = AutoModel.from_pretrained(text_params["model"])
        self.text_model.train()

        pretrained = video_params["pretrained"]
        if video_params["model"] == "SpaceTimeTransformer":
            num_frames = video_params.get("num_frames", 4)
            time_init = video_params.get("time_init", "zeros")
            attention_style = video_params.get("attention_style", "frozen-in-time")
            arch_config = video_params.get("arch_config", "base_patch16_224")
            vit_init = video_params.get("vit_init", "imagenet-21k")
            if arch_config == "base_patch16_224":
                # you can download the checkpoint via wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
                # vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=pretrained)
                vit_model = torch.load(
                    "pretrained/jx_vit_base_p16_224-80ecf9dd.pth", map_location="cpu"
                )
                model = SpaceTimeTransformer(
                    num_frames=num_frames,
                    time_init=time_init,
                    attention_style=attention_style,
                )
            else:
                raise NotImplementedError

            model.head = nn.Identity()
            model.pre_logits = nn.Identity()
            ftr_dim = model.embed_dim
            if load_checkpoint in ["", None]:
                # vit_checkpoint = vit_model.state_dict()
                # model.load_state_dict(vit_checkpoint, strict=False)
                vit_checkpoint = vit_model
                new_vit_dict = state_dict_data_parallel_fix(
                    vit_checkpoint, model.state_dict()
                )
                model.load_state_dict(new_vit_dict, strict=False)
            self.video_model = model
        else:
            raise NotImplementedError(f"{video_params['model']} not implemented")

        # for backwards compatibility (old models)
        self.video_model.fc = nn.Identity()

        # Project to a common embedding
        if projection == "minimal":
            txt_proj = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.text_model.config.hidden_size, projection_dim),
            )

            vid_proj = nn.Sequential(nn.Linear(ftr_dim, projection_dim))
        elif projection == "":
            txt_proj = nn.Identity()
            vid_proj = nn.Identity()
        else:
            raise NotImplementedError
        self.txt_proj = txt_proj
        self.vid_proj = vid_proj

        if load_checkpoint not in ["", None]:
            # # checkpoint = torch.load(load_checkpoint)
            # local_rank = int(os.environ["LOCAL_RANK"])  # fixed by qinghong.
            # checkpoint = torch.load(
            #     load_checkpoint, map_location="cuda:{}".format(local_rank)
            # )
            # state_dict = checkpoint["state_dict"]

            # checkpoint = torch.load(load_checkpoint)
            # local_rank = int(os.environ['LOCAL_RANK'])  # fixed by qinghong.
            # checkpoint = torch.load(load_checkpoint, map_location='cuda:{}'.format(local_rank))
            checkpoint = torch.load(load_checkpoint, map_location="cpu")
            if "state_dict_v" in checkpoint:
                state_dict = checkpoint["state_dict_v"]
            else:
                state_dict = checkpoint["state_dict"]

            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            new_state_dict = self._inflate_positional_embeds(new_state_dict)
            self.load_state_dict(new_state_dict, strict=True)

    def set_device(self, device):
        self.device = device

    def forward(self, data, video_only=False, return_embeds=True):
        if video_only:
            video_data = data["video"]
            video_embeddings = self.compute_video(video_data)
            return video_embeddings

        text_data = data["text"]
        video_data = data["video"]

        text_embeddings = self.compute_text(text_data)
        video_embeddings = self.compute_video(video_data)

        if return_embeds:
            return text_embeddings, video_embeddings

        return sim_matrix(text_embeddings, video_embeddings)

    def compute_text(self, text_data):
        if self.text_params["model"].startswith("bert"):
            text_embeddings = self.text_model(
                text_data["input_ids"], attention_mask=text_data["attention_mask"]
            )["pooler_output"]
        elif self.text_params["model"].startswith("distilbert"):
            text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
        else:
            raise NotImplementedError
        text_embeddings = self.txt_proj(text_embeddings)
        return text_embeddings

    def compute_text_tokens(self, text_data):
        if self.text_params["model"].startswith("bert"):
            text_embeddings = self.text_model(
                text_data["input_ids"], attention_mask=text_data["attention_mask"]
            )["pooler_output"]  # not implement for bert
        elif self.text_params["model"].startswith("distilbert"):
            text_embeddings = self.text_model(**text_data).last_hidden_state
        else:
            raise NotImplementedError

        text_embeddings = self.txt_proj(text_embeddings)
        return text_embeddings

    def compute_video(self, video_data):
        video_embeddings = self.video_model(video_data)
        video_embeddings = self.vid_proj(video_embeddings)
        return video_embeddings

    def _inflate_positional_embeds(self, new_state_dict):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if (
            "video_model.temporal_embed" in new_state_dict
            and "video_model.temporal_embed" in curr_keys
        ):
            load_temporal_embed = new_state_dict["video_model.temporal_embed"]
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.video_params["num_frames"]
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    print(
                        f'### loaded {self.video_params["model"]} model has MORE frames than current...'
                        f'### loading weights, filling in the extras via {self.load_temporal_fix}'
                    )
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    print(
                        f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
                        f'### loading weights, filling in the extras via {self.load_temporal_fix}'
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
                new_state_dict["video_model.temporal_embed"] = new_temporal_embed
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


#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


class ASE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.audio_encoder = AudioEncoder(config)
        self.text_encoder = TextEncoder(config)

        # settings for projection layers
        embed_size = config["embed_size"]
        audio_width = self.audio_encoder.audio_width
        text_width = self.text_encoder.text_width

        self.audio_proj = nn.Sequential(
            nn.Linear(audio_width, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_width, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
        )

        self.temp = nn.Parameter(torch.ones([]) * config["temp"])

        self.embed_reg = config["embed_regularization"]

        self.atc_loss = AudioTextContrastiveLoss()

    def encode_audio(self, audio):
        # if audio.dim() == 3:
        #     # if True:
        #     audio_embeds_tot = []
        #     for idx in range(0, 5):
        #         new_data = audio[:, idx, :].clone().detach()
        #         audio_feats = self.audio_encoder(new_data)
        #         audio_embeds = F.normalize(self.audio_proj(audio_feats), dim=-1)
        #         audio_embeds_tot.append(audio_embeds.detach())
        #         del new_data
        #     audio_embeds = torch.cat(audio_embeds_tot, dim=-2)
        #     # print(audio_embeds)
        # else:
        audio_feats = self.audio_encoder(audio)
        audio_embeds = F.normalize(self.audio_proj(audio_feats), dim=-1)
        # print(audio_embeds)
        return audio_embeds

    def encode_text(self, text):
        text_feats = self.text_encoder(text)
        text_embeds = F.normalize(self.text_proj(text_feats[:, 0, :]), dim=-1)
        return text_embeds

    def forward(self, audio, text, idx):
        audio_embeds = self.encode_audio(audio)
        text_embeds = self.encode_text(text)

        idx = idx.view(-1, 1)
        pos_idx = torch.eq(idx, idx.t()).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        sim_a2t = audio_embeds @ text_embeds.t() / self.temp
        sim_t2a = text_embeds @ audio_embeds.t() / self.temp
        loss = self.atc_loss(sim_a2t, sim_t2a, sim_targets)
        if self.embed_reg:
            loss = (
                loss
                + torch.mean(torch.abs(audio_embeds))
                / torch.sqrt(torch.sum(audio_embeds**2))
                + torch.mean(torch.abs(text_embeds))
                / torch.sqrt(torch.sum(text_embeds**2))
            )

        return loss


def sim_matrix(a, b, eps=1e-8, norm: str = ""):
    """
    added eps for numerical stability
    """
    if a.ndim == 3 == b.ndim:
        a_n, b_n = a.norm(dim=-1)[:, :, None], b.norm(dim=-1)[:, :, None]
    else:
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    if a_norm.ndim == 3 == b_norm.ndim:
        sim_mt = torch.bmm(a_norm, b_norm.transpose(1, 2))
        sim_mt = sim_mt.squeeze(1)
    else:
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    if norm != "":
        if norm == "max":
            sim_mt = F.normalize(sim_mt, dim=1)
        elif norm == "one":
            sim_mt = (sim_mt.T / (sim_mt.sum(dim=1))).T
    return sim_mt.detach()


if __name__ == "__main__":
    pass
