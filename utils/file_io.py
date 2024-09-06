import os
import pickle
from pathlib import Path
from typing import Any, Union

from torch import nn
from typeguard import typechecked


@typechecked
def get_ckpt_name(args: Any, config) -> str:
    if "wavcap" in args.config:
        ckpt = "wavcaps_pt"
        if args.load_ckpt_aud:
            if "Clotho" in args.load_ckpt_aud:
                ckpt = "wavcaps_clotho"
            elif "AudioCaps" in args.load_ckpt_aud:
                ckpt = "wavcaps_audiocaps"
        else:
            if "Clotho" in config._config["arch"]["args"]["load_checkpoint"]:
                ckpt = "wavcaps_clotho"
            elif "AudioCaps" in args.load_ckpt_aud:
                ckpt = "wavcaps_audiocaps"
    else:
        ckpt = "laion"
    return ckpt


@typechecked
def write_res_file(
    both: bool,
    output: str,
    args: Any,
    config,
    use_gpt: bool,
    seed: int,
    folder: str = "folder_results_table1",
    rel_mat: str = "",
) -> None:
    if both is True:
        model_used = "both"
    else:
        model_used = "audio" if "aud_params" in config["arch"]["args"] else "video"
    ckpt_for_name_save = get_ckpt_name(args, config)
    right_sec = (
        config._config["data_loader"]["args"]
        .get("aud_params", {"right_sec": 0})
        .get("right_sec", 0)
    )
    left_sec = (
        config._config["data_loader"]["args"]
        .get("aud_params", {"left_sec": 0})
        .get("left_sec", 0)
    )
    if rel_mat != "":
        if "relevancy_score_epicsounds_mainclasses_" in rel_mat:
            relmatrix = f"_relmatrix_{rel_mat.split('relevancy_score_epicsounds_mainclasses_')[1].rsplit('.json')[0]}"
        else:
            relmatrix = f"_relmatrix_{rel_mat}"
    else:
        relmatrix = ""
    if not os.path.exists(f"{folder}"):
        os.mkdir(f"{folder}")
    with open(
        f'{folder}/results_for_both_{both}_usegpt_{use_gpt}_relevancytype_{config._config["data_loader"]["args"]["relevancy_type"]}_suffix{config._config["data_loader"]["args"]["suffix"]}_modelused{model_used}_ckpt_{ckpt_for_name_save}_rightsec_{right_sec}_leftsec_{left_sec}_seed_{seed}{relmatrix}.txt',
        "w",
    ) as f:
        for entry in output:
            f.write(entry)
        f.write("\n")
        f.write(str(config._config))


def save_results(
    config,
    args,
    res,
    model_used: str,
    metric_name: str,
    use_gpt: bool = False,
    test_file: str = "Unknown",
    vtoa_ratio: int = 1,
    normalisation: str = "",
    model_v: nn.Module = None,
    seed: int = 0,
    right_sec: Union[float, int] = 0,
    left_sec: Union[float, int] = 0,
) -> None:
    test_file = test_file.rsplit(".json", 1)[0]
    ckpt = get_ckpt_name(args, config)
    if normalisation == "":
        normalisation = "None"
    if not Path("egoclip_res_test").exists():
        Path("egoclip_res_test").mkdir(exist_ok=True, parents=True)
    with open(
        f"egoclip_res_test/vtoa_{vtoa_ratio}_normalisation_{normalisation}_modelused_{model_used}_usegpt_{use_gpt}_metric_{metric_name}_testfile_{test_file}_ckpt_{ckpt}_rightsec_{right_sec}_leftsec_{left_sec}_seed_{seed}.pkl",
        "wb",
    ) as f:
        pickle.dump(res, f)
