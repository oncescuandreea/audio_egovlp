import os

import torch
import yaml
from src.open_clip import create_model

from model.model import ASE


def load_model_clap(config, device, args):
    amodel = config["arch"]["args"]["aud_params"]["amodel"].replace("/", "-")
    tmodel = config["arch"]["args"]["text_params"]["tmodel"]
    pretrained = ""
    if tmodel == "bert" or tmodel == "roberta" or tmodel == "bart":
        assert (
            pretrained == "" or args is None
        ), "bert/roberta/bart text encoder does not support pretrained models."

    model, model_cfg = create_model(
        amodel,
        tmodel,
        pretrained,
        precision=config["arch"]["args"]["precision"],
        device=device,
        jit=config["arch"]["args"]["torchscript"],
        force_quick_gelu=config["arch"]["args"]["torchscript"],
        openai_model_cache_dir=os.path.expanduser(
            config["arch"]["args"]["openai_model_cache_dir"]
        ),
        skip_params=True,
        pretrained_audio=config["arch"]["args"]["aud_params"]["pretrained-audio"],
        pretrained_text=config["arch"]["args"]["text_params"]["pretrained-text"],
        enable_fusion=config["arch"]["args"]["enable_fusion"],
        fusion_type=config["arch"]["args"]["fusion_type"],
    )

    model_path = config["arch"]["args"]["load_checkpoint"]
    # checkpoint = torch.load(model_path, map_location=device)
    checkpoint = torch.load(model_path, map_location="cpu")

    if "epoch" in checkpoint:
        # resuming a train checkpoint w/ epoch and optimizer state
        start_epoch = checkpoint["epoch"]
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith("module"):
            sd = {k[len("module.") :]: v for k, v in sd.items()}
        model.load_state_dict(sd)
        print(f"=> resuming checkpoint '{model_path}' (epoch {start_epoch})")
    else:
        # loading a bare (model only) checkpoint for fine-tune or evaluation
        model.load_state_dict(checkpoint)
        start_epoch = 0
    start_epoch = 0

    return model, model_cfg


def load_model_ase(config, args):
    with open(
        "/scratch/shared/beegfs/oncescu/coding/libs/pt/egovlp/configs/eval/inference.yaml",
        "r",
    ) as f:
        config_wav = yaml.safe_load(f)
    config_wav["pretrain_path"] = config["arch"]["args"]["load_checkpoint"]
    print(f'Loading ASE model from {config_wav["pretrain_path"]}')
    # config_wav["seed"] = args.seed if not None else config_wav["seed"]
    # print(f'Using seed {config_wav["seed"]}')
    model = ASE(config_wav)
    cp_path = config_wav["pretrain_path"]
    cp = torch.load(cp_path)
    model.load_state_dict(cp["model"])
    return model
