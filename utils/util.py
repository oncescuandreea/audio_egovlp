import functools
import json
import os
import socket
import string
import time
from collections import OrderedDict
from datetime import datetime
from itertools import repeat
from pathlib import Path
from re import sub

import humanize
import numpy as np
import psutil
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import util as util_st
from transformers import GPT2Tokenizer
from typeguard import typechecked

from utils import nDCG

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def replace_nested_dict_item(obj, key, replace_value):
    for k, v in obj.items():
        if isinstance(v, dict):
            obj[k] = replace_nested_dict_item(v, key, replace_value)
    if key in obj:
        obj[key] = replace_value
    return obj


def state_dict_data_parallel_fix(load_state_dict, curr_state_dict):
    load_keys = list(load_state_dict.keys())
    curr_keys = list(curr_state_dict.keys())

    redo_dp = False
    undo_dp = False
    if not curr_keys[0].startswith("module.") and load_keys[0].startswith(
        "module."
    ):  # this
        undo_dp = True
    elif curr_keys[0].startswith("module.") and not load_keys[0].startswith("module."):
        redo_dp = True

    if undo_dp:  # this
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
    elif redo_dp:
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = "module." + k  # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = load_state_dict
    return new_state_dict


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array
    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print("shape,", x.shape)
    if val:
        x = x.flatten()
        print(
            "mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f"
            % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x))
        )


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def read_json(fname):
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def memory_summary():
    vmem = psutil.virtual_memory()
    msg = (
        f">>> Currently using {vmem.percent}% of system memory "
        f"{humanize.naturalsize(vmem.used)}/{humanize.naturalsize(vmem.available)}"
    )
    print(msg)


@functools.lru_cache(maxsize=64, typed=False)
def memcache(path):
    suffix = Path(path).suffix
    print("loading features >>>", end=" ")
    tic = time.time()
    if suffix == ".npy":
        res = np_loader(path)
    else:
        raise ValueError(f"unknown suffix: {suffix} for path {path}")
    print(
        f"[Total: {time.time() - tic:.1f}s] ({socket.gethostname() + ':' + str(path)})"
    )
    return res


def np_loader(np_path, l2norm=False):
    with open(np_path, "rb") as f:
        data = np.load(f, encoding="latin1", allow_pickle=True)
    if isinstance(data, np.ndarray) and data.size == 1:
        data = data[()]  # handle numpy dict storage convnetion
    if l2norm:
        print("L2 normalizing features")
        if isinstance(data, dict):
            for key in data:
                feats_ = data[key]
                feats_ = feats_ / max(np.linalg.norm(feats_), 1e-6)
                data[key] = feats_
        elif data.ndim == 2:
            data_norm = np.linalg.norm(data, axis=1)
            data = data / np.maximum(data_norm.reshape(-1, 1), 1e-6)
        else:
            raise ValueError("unexpected data format {}".format(type(data)))
    return data


#### for audio


def relevance_mask_generator(mask_dict_aud_rel, levels, aud_relevancy_dict, data_entry):
    if aud_relevancy_dict is not None:
        text_relevance = (
            aud_relevancy_dict[data_entry]
            if data_entry in aud_relevancy_dict
            else "high"
        )
    else:
        # make the code work by saying all examples have high importance. number not relevant
        text_relevance = "high"
    for level in levels:
        mask_dict_aud_rel[level].append(True if level == text_relevance else False)
    return mask_dict_aud_rel


def normalize_sentence(sentence: str):
    """
    This is to deal with examples such as, original key is crack egg on side of bowl
    and chatgpt rewrites it as Crack egg on the side of the bowl
    """
    # Tokenize the sentence into words
    tokens = word_tokenize(sentence.lower())

    # Americanize the sentence:

    # Normalize composed words
    tokens = [word.replace("-", " ") for word in tokens]

    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]

    # Remove stopwords (e.g., "the", "on")
    tokens = [word for word in tokens if word not in stop_words]

    # Use lemmas
    # tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return set(tokens)  # Convert list of tokens to a set to remove duplicates


def move_video_data_to_device(data, device):
    if isinstance(data["video"], list):
        data["video"] = [x.to(device) for x in data["video"]]
    else:
        data["video"] = data["video"].to(device)
    return data


def tokenize_and_move_to_gpu(
    text, tokenizer, model_used: str, config_arch_type: str = "CLAP"
):
    """
    Tokenize text. Move to GPU only if video model is used, otherwise
    the model will take care of it.
    model_used can be audio, video, both
    """
    # if config_arch_type == "ASE":
    #     print("wavcapping the text")
    #     text = text_wavcaps(text)
    tokenized_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    if model_used in ["video", "both"]:
        return {key: val.cuda() for key, val in tokenized_text.items()}
    else:
        return tokenized_text


def initialise_nDCG_values(relevancy_matrix):
    vis_k_counts = nDCG.calculate_k_counts(relevancy_matrix)
    txt_k_counts = nDCG.calculate_k_counts(relevancy_matrix.T)

    vis_IDCG = nDCG.calculate_IDCG(relevancy_matrix, vis_k_counts)
    txt_IDCG = nDCG.calculate_IDCG(relevancy_matrix.T, txt_k_counts)

    k_counts_dict = {"v": vis_k_counts, "t": txt_k_counts}
    IDCG_dict = {"v": vis_IDCG, "t": txt_IDCG}

    return IDCG_dict, k_counts_dict


def initialise_jpose_nDCG_values(relevancy_matrix):
    action_IDCG, action_k_values = initialise_nDCG_values(relevancy_matrix)

    dataset = {}
    dataset["action"] = {}
    dataset["action"]["IDCG"] = action_IDCG
    dataset["action"]["k_values"] = action_k_values
    return dataset


@typechecked
def print_mAP(vis_mAP: np.array, txt_mAP: np.array, output: str = "") -> str:
    print(
        "mAP: VT:{:.3f} TV:{:.3f} AVG:{:.3f}".format(
            np.mean(vis_mAP),
            np.mean(txt_mAP),
            (np.mean(vis_mAP) + np.mean(txt_mAP)) / 2,
        )
    )
    output += "mAP: VT:{:.3f} TV:{:.3f} AVG:{:.3f}\n".format(
        np.mean(vis_mAP),
        np.mean(txt_mAP),
        (np.mean(vis_mAP) + np.mean(txt_mAP)) / 2,
    )
    return output


@typechecked
def print_nDCG(vis_nDCG: np.array, txt_nDCG: np.array, output: str) -> str:
    print(
        "nDCG: VT:{:.3f} TV:{:.3f} AVG:{:.3f}".format(
            np.mean(vis_nDCG),
            np.mean(txt_nDCG),
            (np.mean(vis_nDCG) + np.mean(txt_nDCG)) / 2,
        )
    )
    output += "nDCG: VT:{:.3f} TV:{:.3f} AVG:{:.3f}\n".format(
        np.mean(vis_nDCG),
        np.mean(txt_nDCG),
        (np.mean(vis_nDCG) + np.mean(txt_nDCG)) / 2,
    )
    return output


@typechecked
def calculate_from_split(
    results_description: str,
    mask_dict_aud_rel: dict,
    config,
    indexes: list,
    visual_ndcg: np.array,
    text_ndcg: np.array,
    visual_map: np.array,
    text_map: np.array,
    # similarity_matrix_tot=None,
) -> str:
    """
    Calculates and updates the results_description based on given parameters.

    Parameters:
    - results_description (str): Initial description or results string.
    - mask_dict_aud_rel: dict containing keys such as {low, moderate, high} or {classes}. Each value represents a list of True/False entries corresponding to each example. If True, then that example is in the subset represented by the key.
    - visual_ndcg: np.array containing scores for each visual entry
    ...
    [Other parameters explained]

    Returns:
    - str: Updated results description.
    """
    levels = list(mask_dict_aud_rel.keys())
    suffix = config["data_loader"]["args"]["suffix"]
    # with open("/work/oncescu/data/kinetics700/annotations/indexes.pkl", "wb") as f:
    #     pickle.dump({"indexes": indexes}, f)
    if not any(level in suffix for level in levels):
        for level in levels:
            level_samples = mask_dict_aud_rel[level]
            # text_examples = np.array(level_samples)[indexes]
            print_statement = (
                f"For level {level} there are {sum(level_samples)} aud examples."
            )
            print(print_statement)

            new_visual_ndcg, new_txt_nDCG = (
                visual_ndcg[level_samples],
                text_ndcg,
            )

            results_description += print_statement + "\n"
            results_description = print_nDCG(
                new_visual_ndcg, new_txt_nDCG, results_description
            )

            new_visual_map, new_text_map = (
                visual_map[level_samples],
                text_map,
            )

            results_description = print_mAP(
                new_visual_map, new_text_map, results_description
            )

    return results_description


@typechecked
def gen_mask_array(mask_array: list, text_lemmas_set: set, audio_lemmas: list) -> list:
    """
    This generates a list of weights to help with masking the audio information when it might
    not be relevant. The way it checks if the description has audio content in it is by using
    audio-centric words as selected by GPT4. If these are present in the description, then
    the audio should be used.
    """
    audio_lemma_check = 0.0
    for text_lemma in text_lemmas_set:
        if text_lemma in audio_lemmas:
            audio_lemma_check = 1.0
            break
    mask_array.append(audio_lemma_check)
    return mask_array


@typechecked
def util_text_wavcaps(sentence: str) -> str:
    # transform to lower case
    sentence = sentence.lower()

    # remove any forgotten space before punctuation and double space
    sentence = sub(r'\s([,.!?;:"](?:\s|$))', r"\1", sentence).replace("  ", " ")

    # remove punctuations
    # sentence = sub('[,.!?;:\"]', ' ', sentence).replace('  ', ' ')
    sentence = sub('[(,.!?;:|*")]', " ", sentence).replace("  ", " ")
    return sentence


def text_wavcaps(sentence):
    return sentence
    if isinstance(sentence, list):
        wavcap_sentences = []
        for entry in sentence:
            wavcap_sentences.append(util_text_wavcaps(entry))
        return wavcap_sentences
    else:
        return util_text_wavcaps(sentence)


@typechecked
def strhhmmss_tofloats(str_time: str) -> float:
    """
    Function takes a string in the form of hh:mm:ss.ms and returns a float in seconds
    """
    h, m, sms = str_time.split(":")
    s, ms = sms.split(".")
    ms = ms.ljust(3, "0")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def a2t(audio_embs, cap_embs, return_ranks=False):
    # audio to caption retrieval
    num_audios = int(audio_embs.shape[0] / 5)

    ranks = np.zeros(num_audios)
    top1 = np.zeros(num_audios)
    AP10 = np.zeros(num_audios)
    for index in range(num_audios):
        # get query audio
        audio = audio_embs[5 * index]

        # compute scores
        # d = audio @ cap_embs.T
        d = (
            util_st.cos_sim(torch.Tensor(audio), torch.Tensor(cap_embs))
            .squeeze(0)
            .numpy()
        )
        inds = np.argsort(d)[::-1]

        inds_map = []

        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
            if tmp < 10:
                inds_map.append(tmp + 1)
        inds_map = np.sort(np.array(inds_map))
        # calculate average precision
        if len(inds_map) != 0:
            AP10[index] = np.sum((np.arange(1, len(inds_map) + 1) / inds_map)) / 5
        else:
            AP10[index] = 0.0
        ranks[index] = rank
        top1[index] = inds[0]
    # compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    mAP10 = 100.0 * np.sum(AP10) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return r1, r5, r10, r50, medr, meanr, mAP10, ranks, top1
    else:
        return r1, r5, r10, r50, medr, meanr, mAP10


class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()


def save_correctness_of_text(similarity_matrix: np.array):
    # Extract the diagonal (correct text-video similarity scores)
    diagonal_values = np.diag(similarity_matrix)

    # Find the maximum values in each row
    row_max_values = np.max(similarity_matrix, axis=1)

    # Compare diagonal values with row max values
    correct_indices = np.where(diagonal_values == row_max_values)[0]

    # print("Text indexes where the retrieved video is correct:", correct_indices)
    return correct_indices
