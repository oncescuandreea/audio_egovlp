import argparse
import collections
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import transformers
from sacred import Experiment
from tensorboardX import SummaryWriter

import data_loader.data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import utils.visualizer as module_vis
from parse_config import ConfigParser
from trainer import Multi_Trainer_aud_dist
from utils import load_model_ase, load_model_clap
from utils.util import replace_nested_dict_item

ex = Experiment("train")


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


@ex.main
def run(config, args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.debug == "yes":
        setup(rank, world_size)
    # TODO: improve Create identity (do nothing) visualiser?
    if config["visualizer"]["type"] != "":
        visualizer = config.initialize(
            name="visualizer",
            module=module_vis,
            exp_name=config["name"],
            web_dir=config._web_log_dir,
        )
    else:
        visualizer = None

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    torch.cuda.set_device(args.local_rank)

    # if args.world_size > 1:
    if args.master_address != 9339:
        print("DistributedDataParallel")
        # DistributedDataParallel
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="tcp://{}:{}".format(args.master_address, args.master_port),
            rank=args.rank,
            world_size=args.world_size,
        )

    if args.rank == 0:
        print("world_size", args.world_size, flush=True)
        print("local_rank: ", args.local_rank, flush=True)

    # build tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config["arch"]["args"]["text_params"]["model"], TOKENIZERS_PARALLELISM=False
    )

    if args.use_gpt:
        config._config["data_loader"][0]["args"]["aud_params"]["use_gpt"] = (
            False if args.use_gpt == "false" else True
        )
        print(
            f'Overwriting use_gpt using args from {config["data_loader"][0]["args"]["aud_params"]["use_gpt"]} to {args.use_gpt}'
        )
    config.config["arch"]["args"]["load_checkpoint"] = (
        args.load_ckpt_aud or config._config["arch"]["args"]["load_checkpoint"]
    )
    config._config["data_loader"][0]["args"]["val_file"] = (
        args.val_file
        if args.val_file is not None
        else config._config["data_loader"][0]["args"].get("val_file", "egomcq.json")
    )
    config._config["data_loader"][0]["args"]["test_file"] = (
        args.test_file
        if args.test_file is not None
        else config._config["data_loader"][0]["args"].get("test_file", "egomcq.json")
    )
    config._config["trainer"]["seed"] = (
        args.seed
        if args.seed is not None
        else config._config["trainer"].get("seed", "egomcq.json")
    )
    config._config["arch"]["args"]["load_checkpoint"] = (
        args.load_ckpt_aud or config._config["arch"]["args"]["load_checkpoint"]
    )
    # setup data_loader instances
    if "arch_vid" in config._config:
        config._config["data_loader"][0]["args"]["both"] = True
    data_loader, valid_data_loader = init_dataloaders(config, module_data)
    num_train_samples = [x.n_samples for x in data_loader]
    if args.rank == 0:
        print("Train dataset: ", [x.n_samples for x in data_loader], " samples")
        print("Val dataset: ", [x.n_samples for x in valid_data_loader], " samples")
    # build model architecture, then print to console

    if args.sweep == "true":
        config._config["trainer"]["vtoa_ratio"] = int(args.vtoa)
        config._config["trainer"]["normalisation"] = args.norm

    args.learning_rate1 = config["optimizer"]["args"]["lr"]

    if "aud_params" not in config["arch"]["args"]:
        model = config.initialize("arch", module_arch)
    else:
        if config["arch"]["type"] == "ASE":
            model = load_model_ase(config, args)
        elif config["arch"]["type"] == "CLAP":
            model, _ = load_model_clap(config, "cpu", args)

    if "arch_vid" in config._config:
        model_v = config.initialize("arch_vid", module_arch)
        tokenizer_v = transformers.AutoTokenizer.from_pretrained(
            config["arch_vid"]["args"]["text_params"]["model"],
            TOKENIZERS_PARALLELISM=False,
        )
    else:
        model_v = None
        tokenizer_v = None

    # if args.local_rank == 0:
    #     logger.info(model)

    # get function handles of loss and metrics
    loss = config.initialize(name="loss", module=module_loss)
    metrics = [getattr(module_metric, met) for met in config["metrics"]]
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize("optimizer", transformers, trainable_params)
    lr_scheduler = None
    # num_training_steps = (config['trainer']['epochs'] - config['trainer']['start_epoch'] + 1) * num_train_samples[0] // (data_loader[0].batch_size*args.world_size)
    if "lr_scheduler" in config._config:
        num_training_steps = (
            (config["trainer"]["epochs"] - config["trainer"]["start_epoch"] + 1)
            * (num_train_samples[0] // data_loader[0].batch_size)
            * (args.world_size // config["n_gpu"])
        )
        if args.rank == 0:
            print(
                f"args.world_size is {args.world_size}, orig_batch_size_per_node is {data_loader[0].batch_size},\
                \nno_epochs is {config['trainer']['epochs'] - config['trainer']['start_epoch'] + 1} and num_training_steps is {num_training_steps}"
            )
    if "lr_scheduler" in config._config:
        if "get_cosine_schedule_with_warmup" in config._config["lr_scheduler"]["type"]:
            config._config["lr_scheduler"]["args"]["num_training_steps"] = int(
                num_training_steps
            )
        if hasattr(transformers, config._config["lr_scheduler"]["type"]):
            lr_scheduler = config.initialize("lr_scheduler", transformers, optimizer)
        else:
            print("lr scheduler not found")
    print(f"lr_scheduler is {lr_scheduler}")
    if config["trainer"]["neptune"]:
        writer = ex
    else:
        writer = None

    if args.rank == 0:
        writer = SummaryWriter(log_dir=str(config.tf_dir))

    # trainer = Multi_Trainer_dist(args, model, loss, metrics, optimizer,
    #                   config=config,
    #                   data_loader=data_loader,
    #                   valid_data_loader=valid_data_loader,
    #                   lr_scheduler=lr_scheduler,
    #                   visualizer=visualizer,
    #                   writer=writer,
    #                   tokenizer=tokenizer,
    #                   max_samples_per_epoch=config['trainer']['max_samples_per_epoch'])
    trainer = Multi_Trainer_aud_dist(
        args,
        model,
        loss,
        metrics,
        optimizer,
        config=config,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
        visualizer=visualizer,
        writer=writer,
        tokenizer=tokenizer,
        tokenizer_v=tokenizer_v,
        max_samples_per_epoch=config["trainer"]["max_samples_per_epoch"],
        model_v=model_v,
    )

    trainer.train()
    if args.debug == "yes":
        cleanup()


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
        )
        valid_data_loader = [config.initialize("data_loader", module_data)]
    elif isinstance(config["data_loader"], list):
        data_loader = [
            config.initialize("data_loader", module_data, index=idx)
            for idx in range(len(config["data_loader"]))
        ]
        new_cfg_li = []
        for dl_cfg in config["data_loader"]:
            dl_cfg["args"] = replace_nested_dict_item(dl_cfg["args"], "split", "val")
            dl_cfg["args"] = replace_nested_dict_item(dl_cfg["args"], "batch_size", 1)
            new_cfg_li.append(dl_cfg)
        config._config["data_loader"] = new_cfg_li
        valid_data_loader = [
            config.initialize("data_loader", module_data, index=idx)
            for idx in range(len(config["data_loader"]))
        ]
    else:
        raise ValueError("Check data_loader config, not correct format.")

    return data_loader, valid_data_loader


def find_params_value(file, key):
    # find value of params in params_file
    with open(file, "r") as f:
        for line in f:
            if key + ": " in line:
                return line.split(": ")[1].strip()
    return None


if __name__ == "__main__":
    try:  # with ddp
        master_address = os.environ["MASTER_ADDR"]
        master_port = int(os.environ["MASTER_PORT"])
        world_size = int(os.environ["WORLD_SIZE"])
        try:
            rank = int(os.environ["SLURM_PROCID"])
            local_rank = rank % torch.cuda.device_count()
        except:
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
    except:  # for debug only
        master_address = 9339
        master_port = 1
        world_size = 1
        rank = 0
        local_rank = 0

    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default="configs/pt/egoclip.json",
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o", "--observe", action="store_true", help="Whether to observe (neptune)"
    )
    args.add_argument(
        "-l",
        "--launcher",
        choices=["none", "pytorch"],
        default="none",
        help="job launcher",
    )
    args.add_argument("-k", "--local_rank", type=int, default=local_rank)

    args.add_argument("-ma", "--master_address", default=master_address)
    args.add_argument("-mp", "--master_port", type=int, default=master_port)
    args.add_argument("-ws", "--world_size", type=int, default=world_size)
    args.add_argument("-rk", "--rank", type=int, default=rank)
    args.add_argument("-lr1", "--learning_rate1", type=float, default=2e-4)
    args.add_argument("-sc", "--schedule", default=[60, 80])
    args.add_argument("--norm", default=None)
    args.add_argument("--vtoa", default=10)
    args.add_argument("--sweep", default="false")
    args.add_argument("--seed", default=0, type=int)
    args.add_argument("--load_ckpt_aud", default=None)
    args.add_argument("--use_gpt", default=None)
    args.add_argument("--test_file", default=None)
    args.add_argument("--val_file", default=None)
    args.add_argument("--debug", default="no")

    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(
            ["--lr", "--learning_rate"], type=float, target=("optimizer", "args", "lr")
        ),
        CustomArgs(
            ["--bs", "--batch_size"],
            type=int,
            target=("data_loader", "args", "batch_size"),
        ),
    ]
    config = ConfigParser(args, options)
    args = args.parse_args()
    ex.add_config(config._config)

    if args.rank == 0:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    print("The rank(local) of this node is {}({})".format(args.rank, args.local_rank))

    if config["trainer"]["neptune"]:
        # delete this error if you have added your own neptune credentials neptune.ai
        raise ValueError("Neptune credentials not set up yet.")
        ex.observers.append(
            NeptuneObserver(
                api_token="INSERT TOKEN", project_name="INSERT PROJECT NAME"
            )
        )
        ex.run()
    else:
        run(config, args)
