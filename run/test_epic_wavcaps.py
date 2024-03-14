# this is the main script now that should work for both audio and video
import argparse
import json
import os
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import numpy.ma as ma
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
import transformers
from sacred import Experiment

import data_loader.data_loader as module_data
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils import (calculate_from_split, gen_mask_array,
                   initialise_jpose_nDCG_values, load_model_ase,
                   load_model_clap, mAP, move_video_data_to_device, nDCG,
                   normalize_sentence, print_mAP, print_nDCG,
                   relevance_mask_generator, save_correctness_of_text,
                   text_wavcaps, tokenize_and_move_to_gpu, write_res_file)

ex = Experiment("test")


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_model_embds(
    both,
    data_loader,
    data_key,
    device,
    tokenizer,
    model,
    model_v,
    tokenizer_v,
    audio_lemmas,
    aud_relevancy_dict,
    debug=False,
):
    text_embed_arr = []
    text_embed_arr_vid = []
    vid_embed_arr = []
    text_embed_arr_aud = []
    aud_embed_arr = []
    mask_array = []

    mask_type = "class"
    if mask_type == "relevance":
        mask_dict_aud_rel = {
            "low": [],
            "moderate": [],
            "high": [],
        }  # dictionary keeping a mask for low, moderate and high audio relevance a->t and t->a
        levels = list(mask_dict_aud_rel.keys())
    else:
        all_classes = set()  # Contains all unique classes seen.
        current_classes = []  # Contains class of each example in order.

    len_dataloader = len(data_loader)

    aud_wavs = []
    descriptions = []
    with torch.no_grad():
        if both is False:
            print("got to line 82")
            model_used = "audio" if "aud_params" in config["arch"]["args"] else "video"
            print(f"Model used is {model_used}")
            assert (
                data_key == "text" if model_used == "video" else True
            ), "Video model has not been set up to take "
            # There is only one model used, either video or audio
            for data in tqdm.tqdm(data_loader, total=len_dataloader):
                data_text_orig = data[data_key].copy()
                descriptions.append(data[data_key])
                for data_entry in data["text"]:
                    if mask_type == "relevance":
                        mask_dict_aud_rel = relevance_mask_generator(
                            mask_dict_aud_rel, levels, aud_relevancy_dict, data_entry
                        )
                    else:
                        current_class = data_entry
                        all_classes.add(current_class)
                        current_classes.append(current_class)
                data = move_video_data_to_device(data, device)

                data[data_key] = tokenize_and_move_to_gpu(
                    data[data_key],
                    tokenizer,
                    model_used,
                    config["arch"]["type"],
                )

                if model_used == "audio":
                    if debug is True:
                        warnings.warn("Warning. Debug mode is on. Are you sure?")
                        vid_embed_arr.append(torch.rand(data_loader.batch_size, 1024))
                        text_embed_arr.append(torch.rand(data_loader.batch_size, 1024))
                        continue
                    if config["arch"]["type"] == "CLAP":
                        text_embed, aud_embed = model(
                            data["audio"],
                            data[data_key],
                            device=device,
                            return_embeds=True,
                        )
                    elif config["arch"]["type"] == "ASE":
                        aud_wavs.append(data["audio"]["waveform"].unsqueeze(0))
                        data["audio"]["waveform"] = data["audio"]["waveform"].to(device)
                        aud_embed = model.encode_audio(data["audio"]["waveform"])
                        text_embed = model.encode_text(text_wavcaps(data_text_orig[0]))
                    vid_embed_arr.append(aud_embed.cpu().detach())
                    text_embed_arr.append(text_embed.cpu().detach())
                else:
                    text_embed, vid_embed = model(data, return_embeds=True)
                    vid_embed_arr.append(vid_embed.cpu().detach())
                    text_embed_arr.append(text_embed.cpu().detach())
                del data_text_orig
        else:
            assert (
                tokenizer_v is not None
            ), "Video tokenizer not defined although both models should be used"
            assert (
                model_v is not None
            ), "Video model is required for this particular implementation given that both is True"
            for data in tqdm.tqdm(data_loader, total=len_dataloader):
                for data_entry in data["text"]:
                    if mask_type == "relevance":
                        mask_dict_aud_rel = relevance_mask_generator(
                            mask_dict_aud_rel, levels, aud_relevancy_dict, data_entry
                        )
                text_lemmas_set = normalize_sentence(data["text"][0])
                data_text_orig = data[data_key].copy()
                data = move_video_data_to_device(data, device)
                data_text_aud = tokenize_and_move_to_gpu(
                    data[data_key].copy(), tokenizer, "audio"
                )
                data["text"] = tokenize_and_move_to_gpu(
                    data["text"], tokenizer_v, "video"
                )

                # check if any of the words is in audio based lemmas list
                mask_array = gen_mask_array(mask_array, text_lemmas_set, audio_lemmas)

                # for the audio model
                if config["arch"]["type"] == "CLAP":
                    text_embed_aud, aud_embed = model(
                        data["audio"], data_text_aud, device=device, return_embeds=True
                    )
                elif config["arch"]["type"] == "ASE":
                    data["audio"]["waveform"] = data["audio"]["waveform"].to(device)
                    aud_embed = model.encode_audio(data["audio"]["waveform"])
                    text_embed_aud = model.encode_text(data_text_orig)
                aud_embed_arr.append(aud_embed.cpu().detach())
                text_embed_arr_aud.append(text_embed_aud.cpu().detach())

                # for the video model
                text_embed_vid, vid_embed = model_v(data, return_embeds=True)
                vid_embed_arr.append(vid_embed.cpu().detach())
                text_embed_arr_vid.append(text_embed_vid.cpu().detach())
                del data_text_orig
            print(f"There are {(np.array(mask_array) == 1.0).sum()} audios used")

    if mask_type == "class":
        mask_dict_aud_rel = {}
        for class_name in all_classes:
            mask_dict_aud_rel[class_name] = [
                item_cls == class_name for item_cls in current_classes
            ]

    vid_embeds = torch.cat(vid_embed_arr)
    if both:
        text_embeds_vid = torch.cat(text_embed_arr_vid)
        aud_embeds = torch.cat(aud_embed_arr)
        text_embeds_aud = torch.cat(text_embed_arr_aud)
        return [
            vid_embeds,
            text_embeds_vid,
            aud_embeds,
            text_embeds_aud,
            mask_array,
            mask_dict_aud_rel,
        ]
    else:
        text_embeds = torch.cat(text_embed_arr)
        return [
            vid_embeds,
            text_embeds,
            mask_dict_aud_rel,
            aud_wavs,
            descriptions,
        ]


@ex.main
def run():
    # if True:
    #     setup(rank, world_size)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    ### Overwriting config content based on arguments for evaluation purposes
    config._config["data_loader"]["args"].update(
        {
            "split": args.split,
            "tsfm_split": "test",
            "shuffle": False,
            "batch_size": args.batch_size,
            "sliding_window_stride": args.sliding_window_stride,
            "relevancy_type": args.relevancy_type
            if args.relevancy_type is not None
            else config._config["data_loader"]["args"].get("relevancy_type", "caption"),
            "suffix": args.suffix
            if args.suffix is not None
            else config._config["data_loader"]["args"].get("suffix", ""),
            "val_test_split": args.val_test_split
            if args.val_test_split is not None
            else config._config["data_loader"]["args"].get("val_test_split", "test"),
        }
    )

    config._config["arch"]["args"]["load_checkpoint"] = (
        args.load_ckpt_aud or config._config["arch"]["args"]["load_checkpoint"]
    )
    if "aud_params" in config._config["data_loader"]["args"]:
        config._config["data_loader"]["args"]["aud_params"].update(
            {
                "right_sec": args.right_sec
                if args.right_sec is not None
                else config._config["data_loader"]["args"]["aud_params"].get(
                    "right_sec", 0
                ),
                "left_sec": args.left_sec
                if args.left_sec is not None
                else config._config["data_loader"]["args"]["aud_params"].get(
                    "left_sec", 0
                ),
            }
        )
    ### Finished overwriting classic config settings

    both = False if "arch_vid" not in config._config else True
    if both is True:
        config._config["data_loader"]["args"]["both"] = both
    data_loader = config.initialize("data_loader", module_data)

    # build tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config["arch"]["args"]["text_params"]["model"], TOKENIZERS_PARALLELISM=False
    )
    # build model architecture
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

    if both:
        with open(
            "/scratch/shared/beegfs/oncescu/data/epic-kitchens-100-annotations/retrieval_annotations/audio_lemmas.pkl",
            "rb",
        ) as f:
            audio_lemmas = pickle.load(f)

    if "aud_params" in config["arch"]["args"] and config["arch"]["type"] == "CLAP":
        print("Comparing weights sum values:\n")
        print(
            f"model.audio_branch.layers[0].blocks[0].mlp.fc1.weight.sum() is {model.audio_branch.layers[0].blocks[0].mlp.fc1.weight.sum()}"
        )

    model = torch.nn.DataParallel(model) if config["n_gpu"] > 1 else model
    model_v = (
        torch.nn.DataParallel(model_v)
        if config["n_gpu"] > 1 and model_v is not None
        else model_v
    )

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()
    model_v = model_v.to(device).eval() if model_v is not None else model_v

    use_gpt = config["data_loader"]["args"].get("aud_params", {}).get("use_gpt", False)
    if args.use_gpt:
        print(f"Overwriting use_gpt using args from {use_gpt} to {args.use_gpt}")
        use_gpt = False if args.use_gpt == "false" else True
        config._config["data_loader"]["args"]["aud_params"]["use_gpt"] = use_gpt

    data_key = "text_gpt" if use_gpt else "text"

    path_dataframes = config["data_loader"]["args"]["meta_dir"]

    if args.aud_relevancy_name != "":
        with open(
            Path(path_dataframes) / args.aud_relevancy_name,
            "r",
        ) as f:
            aud_relevancy_dict = json.load(f)
    else:
        aud_relevancy_dict = None

    list_eval_results = get_model_embds(
        both,
        data_loader,
        data_key,
        device,
        tokenizer,
        model=model,
        model_v=model_v,
        tokenizer_v=tokenizer_v,
        audio_lemmas=audio_lemmas if both else None,
        aud_relevancy_dict=aud_relevancy_dict,
    )
    if both:
        (
            vid_embeds,
            text_embeds_vid,
            aud_embeds,
            text_embeds_aud,
            mask_array,
            mask_dict_aud_rel,
        ) = list_eval_results
        # (
        #     _,
        #     _,
        #     _,
        #     _,
        #     mask_array,
        #     mask_dict_aud_rel,
        # ) = list_eval_results
    else:
        (
            vid_embeds,
            text_embeds,
            mask_dict_aud_rel,
            aud_wavs,
            descriptions,
        ) = list_eval_results

    if "50" in config._config["data_loader"]["args"]["suffix"]:
        if not both:
            print(f"vid: {vid_embeds[0]}")
            print(f"text: {text_embeds[0]}")
            with open("dict_embeds.pkl", "wb") as f:
                pickle.dump({"vid": vid_embeds, "text": text_embeds}, f)
        else:
            print(f"vid: {vid_embeds[0]}")
            print(f"text_vid: {text_embeds_vid[0]}")
            print(f"aud: {aud_embeds[0]}")
            print(f"text_aud: {text_embeds_aud[0]}")
            with open("dict_embeds_both.pkl", "wb") as f:
                pickle.dump(
                    {
                        "vid": vid_embeds,
                        "text_vid": text_embeds_vid,
                        "aud": aud_embeds,
                        "text_aud": text_embeds_aud,
                    },
                    f,
                )

    if use_gpt:
        dataset_target_split = data_loader.dataset.target_split_fp.split(".csv")[0]
        dataset_target_split = f"{dataset_target_split}_gpt.csv"
        dataset_target_split_sentence = (
            data_loader.dataset.target_split_sentence_fp.split(".csv")[0]
        )
        dataset_target_split_sentence = f"{dataset_target_split_sentence}_gpt.csv"
        path_relevancy = data_loader.dataset.path_relevancy.split(".pkl")[0]
        if os.path.exists(f"{path_relevancy}_gpt.pkl") is True:
            path_relevancy = f"{path_relevancy}_gpt.pkl"
        else:
            path_relevancy = data_loader.dataset.path_relevancy
    else:
        dataset_target_split = data_loader.dataset.target_split_fp
        dataset_target_split_sentence = data_loader.dataset.target_split_sentence_fp
        path_relevancy = data_loader.dataset.path_relevancy
        print(f"dataset_target_split_sentence is {dataset_target_split_sentence}")
        print(f"dataset_target_split is {dataset_target_split}")
        print(f"path_relevancy is {path_relevancy}")
    video_id = pd.read_csv(os.path.join(path_dataframes, dataset_target_split)).values[
        :, 0
    ]
    print(f"Path for video_id is {dataset_target_split}")
    text_id = pd.read_csv(
        os.path.join(path_dataframes, dataset_target_split_sentence)
    ).values[:, 0]
    print(f"Path for text_id is {dataset_target_split_sentence}")

    # some descriptions are repetitive so need to find correspondance between the two csv files
    # find video index corresponding to sentence
    # len(text_id) <= len(video_id), one sentence can correspond to multiple videos.
    # Generate list of len(text_id) called index to keep in mind
    indexes = []
    for elem in text_id:
        indexes.append(video_id.tolist().index(elem))

    # load matrix of relevancy

    print(f"Path relevancy used is {path_relevancy}")
    pkl_file = open(path_relevancy, "rb")
    # relevancy has shape no_vid x no_descr and depending on how many joint nouns and verbs
    # there are values can be between 0. and 1.
    relevancy = pickle.load(pkl_file)
    dataset = initialise_jpose_nDCG_values(relevancy)

    # Uncomment line below if you want to save embeddings and sim matrix for future use
    # save_extra_information(text_embeds, vid_embeds, aud_wavs)

    if args.dual_softmax == "False":
        print("Not using dual softmax")
        if both:
            # similarity_matrix_vid = (sim_matrix(text_embeds_vid, vid_embeds) + 1)/2
            # similarity_matrix_aud = (sim_matrix(text_embeds_aud, aud_embeds) + 1)/2
            print("Got here")
            if (
                "low" in config._config["data_loader"]["args"]["suffix"]
                or "moderate" in config._config["data_loader"]["args"]["suffix"]
                or "high" in config._config["data_loader"]["args"]["suffix"]
                or "collision" in config._config["data_loader"]["args"]["suffix"]
                or "tail" in config._config["data_loader"]["args"]["suffix"]
            ):
                idx_versions = [1]
            else:
                # idx_versions = [1, 8, 9, 10]
                idx_versions = [1]
            output = ""
            for idx in idx_versions:
                output = module_metric.sim_matrices_both(
                    text_embeds_aud,
                    text_embeds_vid,
                    vid_embeds,
                    aud_embeds,
                    indexes,
                    config,
                    relevancy,
                    dataset,
                    f"v{idx}",
                    mask_array,
                    mask_dict_aud_rel,
                    output=output,
                )
            output += args.config
            write_res_file(
                both,
                output,
                args,
                config,
                use_gpt,
                seed=args.seed,
                folder=args.folder,
                rel_mat=args.aud_relevancy_name,
            )
            return
        else:
            print("Got here for video line 507")
            (
                similarity_matrix,
                similarity_matrix2,
                similarity_matrix2_t,
                similarity_matrix3,
                similarity_matrix3_t,
                similarity_matrix_tot,  # n_text*n_vid
            ) = module_metric.sim_matrices_vid_only(text_embeds, vid_embeds, indexes)

    else:
        print("Using dual softmax")
        if both:
            similarity_matrix_vid = module_metric.sim_matrix_mm(
                text_embeds_vid, vid_embeds
            )
            # similarity_matrix_vid = (
            #     module_metric.softmax_numpy(similarity_matrix_vid / 500, dim=1)
            #     * similarity_matrix_vid
            # )
            # similarity_matrix_vid = module_metric.softmax_numpy(
            #     similarity_matrix_vid, dim=0
            # )

            similarity_matrix_aud = module_metric.sim_matrix_mm(
                text_embeds_aud, aud_embeds
            )
            # similarity_matrix_aud = (
            #     module_metric.softmax_numpy(similarity_matrix_aud / 500, dim=1)
            #     * similarity_matrix_aud
            # )
            # similarity_matrix_aud = module_metric.softmax_numpy(
            #     similarity_matrix_aud, dim=0
            # )

            similarity_matrix = (similarity_matrix_vid + similarity_matrix_aud) / 2
            similarity_matrix = (
                module_metric.softmax_numpy(similarity_matrix / 500, dim=1)
                * similarity_matrix
            )
            similarity_matrix = module_metric.softmax_numpy(similarity_matrix, dim=0)

            # similarity_matrix = (similarity_matrix_vid + similarity_matrix_aud) / 2
        else:
            similarity_matrix = module_metric.sim_matrix_mm(text_embeds, vid_embeds)
            similarity_matrix = (
                module_metric.softmax_numpy(similarity_matrix / 500, dim=1)
                * similarity_matrix
            )
            similarity_matrix = module_metric.softmax_numpy(similarity_matrix, dim=0)
        similarity_matrix_tot = similarity_matrix.copy()
        similarity_matrix = similarity_matrix.T[:, indexes]
        similarity_matrix2 = similarity_matrix
        similarity_matrix2 = similarity_matrix2.T[:, indexes]
        similarity_matrix2_t = similarity_matrix2.T
        similarity_matrix3 = similarity_matrix2
        similarity_matrix3_t = similarity_matrix3.T

    # sim_matrix_name = (
    #     "similarity_matrix.pkl" if use_gpt is False else "similarity_matrix_gpt.pkl"
    # )
    # with open(sim_matrix_name, "wb") as f:
    #     pickle.dump(similarity_matrix, f)

    correct_indices = set(save_correctness_of_text(similarity_matrix_tot))
    with open(f"descriptions_{data_key}.txt", "w") as f:
        for idx, description in enumerate(descriptions):
            if idx in correct_indices:
                f.write(f"{description}: Correct\n")
            else:
                f.write(f"{description}: Incorrect\n")

    visual_ndcg = nDCG.calculate_nDCG(
        similarity_matrix,
        relevancy,
        dataset["action"]["k_values"]["v"],
        IDCG=dataset["action"]["IDCG"]["v"],
        reduction=None,
    )
    text_ndcg = nDCG.calculate_nDCG(
        similarity_matrix.T,
        relevancy.T,
        dataset["action"]["k_values"]["t"],
        IDCG=dataset["action"]["IDCG"]["t"],
        reduction=None,
    )
    output = ""

    output = print_nDCG(visual_ndcg, text_ndcg, output)

    visual_map = mAP.calculate_mAP(similarity_matrix, relevancy, reduction=None)
    text_map = mAP.calculate_mAP(similarity_matrix.T, relevancy.T, reduction=None)
    output = print_mAP(visual_map, text_map, output)
    output = calculate_from_split(
        output,
        mask_dict_aud_rel,
        config,
        indexes,
        visual_ndcg,
        text_ndcg,
        visual_map,
        text_map,
        # similarity_matrix_tot if similarity_matrix_tot is not None else None,
    )

    ret_v2t = module_metric.v2t_metrics(similarity_matrix_tot)
    output += f"V2T R@1: {ret_v2t['R1']}, R@5: {ret_v2t['R5']}, R@10: {ret_v2t['R10']}, R@MedR: {ret_v2t['MedR']}, R@MeanR: {ret_v2t['MeanR']}\n"
    print(
        f"V2T R@1: {ret_v2t['R1']}, R@5: {ret_v2t['R5']}, R@10: {ret_v2t['R10']}, R@MedR: {ret_v2t['MedR']}, R@MeanR: {ret_v2t['MeanR']}"
    )

    ret_t2v = module_metric.t2v_metrics(similarity_matrix_tot)
    output += f"T2V R@1: {ret_t2v['R1']}, R@5: {ret_t2v['R5']}, R@10: {ret_t2v['R10']}, R@MedR: {ret_t2v['MedR']}, R@MeanR: {ret_t2v['MeanR']}\n"
    print(
        f"T2V R@1: {ret_t2v['R1']}, R@5: {ret_t2v['R5']}, R@10: {ret_t2v['R10']}, R@MedR: {ret_t2v['MedR']}, R@MeanR: {ret_t2v['MeanR']}"
    )

    output += args.config

    write_res_file(
        both,
        output,
        args,
        config,
        use_gpt,
        seed=args.seed,
        folder=args.folder,
        rel_mat=args.aud_relevancy_name,
    )
    # if True:
    #     cleanup()


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
        "-r",
        "--resume",
        # default='results_egoclip/EgoClip_M_EgoNCE_N_V_Neg_Seg_60/models/0510_10/checkpoint-epoch1.pth'
        # default='results/EgoClip_EPIC_16f_best_rel_01_margin_02/models/0512_01/checkpoint-epoch100.pth',
        #   default='results/EgoClip_EPIC_16f_best_rel_01_margin_04_weight/models/0512_02/checkpoint-epoch100.pth',
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-gpu",
        "--gpu",
        default=0,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-s",
        "--sliding_window_stride",
        default=-1,
        type=int,
        help="test time temporal augmentation, repeat samples with different start times.",
    )
    args.add_argument(
        "--save_feats",
        default="/apdcephfs/private_qinghonglin/",
        help="path to store text & video feats, this is for saving embeddings if you want to do offline retrieval.",
    )
    args.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="split to evaluate on.",
    )
    args.add_argument("--batch_size", default=1, type=int, help="size of batch")
    args.add_argument(
        "--dual_softmax",
        default="True",
        type=str,
        help="whether adopt dual-softmax for inference",
    )
    args.add_argument("--norm", default=None)
    args.add_argument("--vtoa", default=10)
    args.add_argument("--use_gpt", default=None)
    args.add_argument("--suffix", default=None)
    args.add_argument("--suffix_train", default="")
    args.add_argument("--relevancy_type", default=None)
    args.add_argument("--val_test_split", default=None)
    args.add_argument("--right_sec", default=None)
    args.add_argument("--left_sec", default=None)
    args.add_argument("--folder", default="folder_results")
    args.add_argument("-k", "--local_rank", type=int, default=local_rank)
    args.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    args.add_argument("--load_ckpt_aud", default=None)
    args.add_argument("--aud_relevancy_name", type=str, default="")

    args.add_argument("-ma", "--master_address", default=master_address)
    args.add_argument("-mp", "--master_port", type=int, default=master_port)
    args.add_argument("-ws", "--world_size", type=int, default=world_size)
    args.add_argument("-rk", "--rank", type=int, default=rank)

    config = ConfigParser(args, test=True, eval_mode="epic")

    # hack to get sliding into config
    args = args.parse_args()
    config._config["sliding_window_stride"] = args.sliding_window_stride
    ex.add_config(config.config)

    os.environ["CUDA_VISIBLE_DEVICES"] = "" + str(args.gpu)

    ex.run()
