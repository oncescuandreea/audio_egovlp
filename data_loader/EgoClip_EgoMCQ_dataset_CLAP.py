import json
import os
import re
from contextlib import suppress

import librosa
import numpy as np
import pandas as pd
import torchvision

from base.base_dataset import get_video_len, video_reader

try:
    from data_loader.transforms import init_video_transform_dict
except:
    from transforms import init_video_transform_dict

import pickle
import random

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from utils import float32_to_int16, get_mel, int16_to_float32

set_to_rem = [
    "#C C",
    "#c c",
    "# C C",
    "# c c",
    "#C",
    "#c",
    "# C",
    "# c",
    "#O",
    "# O",
    "#o",
    "# X",
    "# x",
    "#x",
    "#X",
    "3C C",
    "C C",
    "#B",
    "# Y",
    "#0",
    "#Person",
    "O A",
]


class EgoClip_EgoMCQ_CLAP(Dataset):
    def __init__(
        self,
        dataset_name,
        text_params,
        aud_params,
        video_params,
        data_dir,
        meta_dir=None,
        split="train",
        tsfms=None,
        cut=None,
        subsample=1,
        sliding_window_stride=-1,
        reader="decord",
        neg_param=None,
        both=False,
        val_file="egomcq_aud_full_filtered.json",
        test_file="egomcq_aud_full_filtered.json",
    ):
        self.dataset_name = dataset_name
        self.text_params = text_params
        # if self.text_params['tmodel'] == "roberta":
        #     from transformers import RobertaTokenizer
        #     self.tokenize = RobertaTokenizer.from_pretrained('roberta-base')
        self.aud_params = aud_params
        self.video_params = video_params
        # check for environment variables
        self.data_dir = os.path.expandvars(data_dir)
        if meta_dir is not None:
            self.meta_dir = os.path.expandvars(meta_dir)
        else:
            self.meta_dir = self.data_dir
        self.split = split
        self.transforms = tsfms
        self.cut = cut
        self.subsample = subsample
        self.sliding_window_stride = sliding_window_stride
        self.video_reader = video_reader[reader]
        self.label_type = "caption"
        self.neg_param = neg_param
        self.both = both
        self.val_file = val_file
        self.test_file = test_file
        self._load_metadata()
        if self.sliding_window_stride != -1:
            if self.split != "test":
                raise ValueError(
                    "Fixing frame sampling is for test time only. can remove but..."
                )
            self._fix_temporal_samples()
        try:
            if self.aud_params["max_length"] != 0:
                self.max_length = (
                    self.aud_params["max_length"] * self.aud_params["sample_rate"]
                )
            else:
                self.max_length = 0
        except KeyError:
            try:
                self.max_length = self.aud_params["max_len"]
            except KeyError:
                raise KeyError("Max_length not defined in any way in config")
        self.right_sec = float(
            self.aud_params["right_sec"] if "right_sec" in self.aud_params else 0
        )
        self.left_sec = float(
            self.aud_params["left_sec"] if "left_sec" in self.aud_params else 0
        )
        print(
            f"Using {self.right_sec} extra seconds to the right and {self.left_sec} extra seconds to the left."
        )
        with open(
            "./dataset/egomcq_aud_orig_to_clean.pkl",
            "rb",
        ) as f:
            self.orig_to_strip = pickle.load(f)

    def _get_video_lens(self):
        vlen_li = []
        for idx, row in self.metadata.iterrows():
            video_path = self._get_video_path(row)[0]
            vlen_li.append(get_video_len(video_path))

        return vlen_li

    def _fix_temporal_samples(self):
        self.metadata["vlen"] = self._get_video_lens()
        self.metadata["frame_intervals"] = self.metadata["vlen"].apply(
            lambda x: np.linspace(
                start=0, stop=x, num=min(x, self.video_params["num_frames"]) + 1
            ).astype(int)
        )
        self.metadata["fix_start"] = self.metadata["frame_intervals"].apply(
            lambda x: np.arange(0, int(x[-1] / len(x - 1)), self.sliding_window_stride)
        )
        self.metadata = self.metadata.explode("fix_start")

    def _load_metadata(self):
        split_files = {
            "train": "egoclip.csv",
            "val": "egomcq_aud_full_filtered.json",  # this is the full val/test set minus duplicates
            "test": "egomcq_aud_full_filtered.json",
        }
        split_files["val"] = self.val_file
        split_files["test"] = self.test_file
        split_files_gpt = {
            "train": "egoclip.csv",
            "val": f"{split_files['val'].rsplit('.json', 1)[0]}_gpt.json",  # come back and add _gpt support
            "test": f"{split_files['test'].rsplit('.json', 1)[0]}_gpt.json",  # come back and add _gpt support
        }
        target_split_fp = split_files[self.split]
        target_split_fp_gpt = split_files_gpt[self.split]
        print(f"Using files {target_split_fp} and {target_split_fp_gpt}")
        self.target_split_fp = target_split_fp
        self.target_split_fp_gpt = target_split_fp_gpt

        self.chunk_sec = 600  # Each segment is up to 600s
        self.noun_dim = 582  # num of nouns of ego4d taxonomy dictionary
        self.verb_dim = 118  # num of verbs of ego4d taxonomy dictionary

        if self.split == "train":
            self.metadata = pd.read_csv(
                os.path.join(self.meta_dir, target_split_fp),
                sep="\t",
                error_bad_lines=False,
                dtype={
                    "video_uid": "str",
                    "video_dur": "str",
                    "narration_source": "str",
                    "narration_ind": "str",
                    "narration_time": "str",
                    "clip_start": "str",
                    "clip_end": "str",
                    "clip_text": "str",
                    "tag_verb": "str",
                    "tag_noun": "str",
                },
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
            if ".csv" in target_split_fp:
                self.metadata = pd.read_csv(
                    os.path.join(self.meta_dir, target_split_fp),
                    sep="\t",
                    error_bad_lines=False,
                    dtype={
                        "video_uid": "str",
                        "video_dur": "str",
                        "narration_source": "str",
                        "narration_ind": "str",
                        "narration_time": "str",
                        "clip_start": "str",
                        "clip_end": "str",
                        "clip_text": "str",
                        "tag_verb": "str",
                        "tag_noun": "str",
                    },
                )
            else:
                with open(os.path.join(self.meta_dir, target_split_fp), "r") as load_f:
                    self.metadata = json.load(load_f)
                with open(
                    os.path.join(self.meta_dir, target_split_fp_gpt), "r"
                ) as load_f:
                    self.metadata_gpt = json.load(load_f)

    def _get_video_path(self, sample):
        video_uid = sample["video_uid"]
        try:
            video_start_sec = max(float(sample["clip_start"]), 0)
            video_end_sec = max(float(sample["clip_end"]), 0)

            chunk_start_id = int(video_start_sec // self.chunk_sec)
            chunk_end_id = int(video_end_sec // self.chunk_sec)

            full_video_start_fp = os.path.join(
                self.data_dir, video_uid, str(chunk_start_id) + ".mp4"
            )
            full_video_end_fp = os.path.join(
                self.data_dir, video_uid, str(chunk_end_id) + ".mp4"
            )

            video_fp = [full_video_start_fp, full_video_end_fp]
            video_sec = [video_start_sec, video_end_sec]
            bound_sec = (chunk_start_id + 1) * self.chunk_sec
            return video_fp, video_sec, bound_sec
        except Exception as e:
            print(f"Exception {e} for sample {sample}")
            return -1, -1, -1

    def _get_video_frames(self, video_fp, video_sec, bound_sec):
        video_loading = self.video_params.get("loading", "strict")
        try:
            if os.path.isfile(video_fp[0]) and os.path.isfile(video_fp[1]):
                imgs, idxs = self.video_reader(
                    video_fp[0],
                    video_fp[1],
                    self.video_params["num_frames"],
                    self.frame_sample,
                    start_sec=video_sec[0],
                    end_sec=video_sec[1],
                    bound_sec=bound_sec,
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
                imgs = torchvision.transforms.ToTensor()(imgs).unsqueeze(0)

        if self.transforms is not None:
            if self.video_params["num_frames"] > 1:
                imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
                imgs = self.transforms(imgs)
                imgs = imgs.transpose(0, 1)  # recover
            else:
                imgs = self.transforms(imgs)

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

    def _get_audio_features(
        self,
        sample,
        audio_data,
        max_len,
        data_truncating,
        data_filling,
        audio_cfg,
        require_grad=False,
    ):
        """
        Calculate and add audio features to sample.
        Sample: a dict containing all the data of current sample.
        audio_data: a tensor of shape (T) containing audio data.
        max_len: the maximum length of audio data.
        data_truncating: the method of truncating data.
        data_filling: the method of filling data.
        audio_cfg: a dict containing audio configuration. Comes from model_cfg['audio_cfg'].
        require_grad: whether to require gradient for audio data.
            This is useful when we want to apply gradient-based classifier-guidance.
        """
        if not isinstance(sample, dict):
            aud_sample = {}
        else:
            aud_sample = sample.copy()
        grad_fn = suppress if require_grad else torch.no_grad
        with grad_fn():
            if len(audio_data) > max_len:
                if data_truncating == "rand_trunc":
                    longer = torch.tensor([True])
                elif data_truncating == "wavcap_trunc":
                    if max_len != 0:
                        max_start = audio_data.shape[-1] - max_len
                        start = random.randint(0, max_start)
                        audio_data = audio_data[start : start + max_len]
                        longer = torch.tensor([True])
                elif data_truncating == "fusion":
                    # fusion
                    mel = get_mel(audio_data, audio_cfg)
                    # split to three parts
                    chunk_frames = (
                        max_len // audio_cfg["hop_size"] + 1
                    )  # the +1 related to how the spectrogram is computed
                    total_frames = mel.shape[0]
                    if chunk_frames == total_frames:
                        # there is a corner case where the audio length is
                        # larger than max_len but smaller than max_len+hop_size.
                        # In this case, we just use the whole audio.
                        mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                        aud_sample["mel_fusion"] = mel_fusion
                        longer = torch.tensor([False])
                    else:
                        ranges = np.array_split(
                            list(range(0, total_frames - chunk_frames + 1)), 3
                        )
                        # print('total_frames-chunk_frames:', total_frames-chunk_frames,
                        #       'len(audio_data):', len(audio_data),
                        #       'chunk_frames:', chunk_frames,
                        #       'total_frames:', total_frames)
                        if len(ranges[1]) == 0:
                            # if the audio is too short, we just use the first chunk
                            ranges[1] = [0]
                        if len(ranges[2]) == 0:
                            # if the audio is too short, we just use the first chunk
                            ranges[2] = [0]
                        # randomly choose index for each part
                        idx_front = np.random.choice(ranges[0])
                        idx_middle = np.random.choice(ranges[1])
                        idx_back = np.random.choice(ranges[2])
                        # select mel
                        mel_chunk_front = mel[idx_front : idx_front + chunk_frames, :]
                        mel_chunk_middle = mel[
                            idx_middle : idx_middle + chunk_frames, :
                        ]
                        mel_chunk_back = mel[idx_back : idx_back + chunk_frames, :]

                        # shrink the mel
                        mel_shrink = torchvision.transforms.Resize(
                            size=[chunk_frames, audio_cfg["mel_bins"]]
                        )(mel[None])[0]
                        # logging.info(f"mel_shrink.shape: {mel_shrink.shape}")

                        # stack
                        mel_fusion = torch.stack(
                            [
                                mel_shrink,
                                mel_chunk_front,
                                mel_chunk_middle,
                                mel_chunk_back,
                            ],
                            dim=0,
                        )
                        aud_sample["mel_fusion"] = mel_fusion
                        longer = torch.tensor([True])
                else:
                    raise NotImplementedError(
                        f"data_truncating {data_truncating} not implemented"
                    )
                # random crop to max_len (for compatibility)
                if data_truncating != "wavcap_trunc":
                    overflow = len(audio_data) - max_len
                    idx = np.random.randint(0, overflow + 1)
                    audio_data = audio_data[idx : idx + max_len]

            else:  # padding if too short
                if len(audio_data) < max_len:  # do nothing if equal
                    if data_filling == "repeatpad":
                        try:
                            n_repeat = int(max_len / len(audio_data))
                        except ZeroDivisionError:
                            print(sample["video_uid"])
                            raise ZeroDivisionError
                        audio_data = audio_data.repeat(n_repeat)
                        # audio_data = audio_data.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                        # audio_data = F.interpolate(audio_data,size=max_len,mode="bicubic")[0,0,0]
                        audio_data = F.pad(
                            audio_data,
                            (0, max_len - len(audio_data)),
                            mode="constant",
                            value=0,
                        )
                    elif data_filling == "pad":
                        audio_data = F.pad(
                            audio_data,
                            (0, max_len - len(audio_data)),
                            mode="constant",
                            value=0,
                        )
                    elif data_filling == "repeat":
                        n_repeat = int(max_len / len(audio_data))
                        audio_data = audio_data.repeat(n_repeat + 1)[:max_len]
                    else:
                        raise NotImplementedError(
                            f"data_filling {data_filling} not implemented"
                        )
                if data_truncating == "fusion":
                    mel = get_mel(audio_data, audio_cfg)
                    mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                    aud_sample["mel_fusion"] = mel_fusion
                longer = torch.tensor([False])

        aud_sample["longer"] = longer
        aud_sample["waveform"] = audio_data

        return aud_sample

    def _get_audio_dict(
        self,
        sample,
        audio_ext,
        max_len,
        audio_cfg,
        video_fp,
        video_sec,
        bound_sec,
        class_index_dict=None,
        data_filling="pad",
        data_truncating="rand_trunc",
        text_augment_selection=None,
    ):
        if os.path.isfile(video_fp[0]) and os.path.isfile(video_fp[1]):
            path_v, folder_v_0, file_v_0 = video_fp[0].rsplit("/", 2)
            path_v, folder_v_1, file_v_1 = video_fp[1].rsplit("/", 2)

            if file_v_0 == file_v_1:
                audio_fp = (
                    "/scratch/shared/beegfs/oncescu/shared-datasets/Ego4D/ego4d_chunked_audio/"
                    + folder_v_0
                    + "/"
                    + file_v_0.rsplit(".mp4", 1)[0]
                    + ".flac"
                )

                if os.path.isfile(audio_fp):
                    duration = video_sec[1] - video_sec[0]
                    offset = video_sec[0] - int(file_v_0[:-4]) * 600.0
                    if (offset + duration + self.right_sec) <= (
                        int(file_v_0[:-4]) + 1
                    ) * 600:
                        duration += self.right_sec
                    if offset - self.left_sec >= int(file_v_0[:-4]) * 600:
                        offset -= self.left_sec
                        duration += self.left_sec
                    audio_data, orig_sr = librosa.load(
                        audio_fp,
                        sr=self.aud_params["sample_rate"],
                        offset=offset,
                        duration=duration,
                        mono=True,
                    )
                    assert (
                        orig_sr == self.aud_params["sample_rate"]
                    ), f"Loading of audio did not go according to plan for {audio_fp}"
                    if self.aud_params["sample_rate"] == 48000:
                        audio_data = int16_to_float32(float32_to_int16(audio_data))
                    audio_data = torch.tensor(audio_data).float()
                else:
                    print(f"ERRRRROOOOR with {audio_fp}")
                    # print(audio_fp)
            else:
                audio_fp_0 = (
                    "/scratch/shared/beegfs/oncescu/shared-datasets/Ego4D/ego4d_chunked_audio/"
                    + folder_v_0
                    + "/"
                    + file_v_0.rsplit(".mp4", 1)[0]
                    + ".flac"
                )
                audio_fp_1 = (
                    "/scratch/shared/beegfs/oncescu/shared-datasets/Ego4D/ego4d_chunked_audio/"
                    + folder_v_1
                    + "/"
                    + file_v_1.rsplit(".mp4", 1)[0]
                    + ".flac"
                )
                if os.path.exists(audio_fp_0) and os.path.exists(audio_fp_1):
                    duration_1 = video_sec[1] - int(file_v_1[:-4]) * 600.0
                    offset_0 = video_sec[0] - int(file_v_0[:-4]) * 600.0
                    if (duration_1 + self.right_sec) <= (int(file_v_1[:-4]) + 1) * 600:
                        duration_1 += self.right_sec
                    if offset_0 - self.left_sec >= int(file_v_0[:-4]) * 600:
                        offset_0 -= self.left_sec
                        duration_1 += self.left_sec
                    audio_data_0, orig_sr0 = librosa.load(
                        audio_fp_0,
                        sr=self.aud_params["sample_rate"],
                        offset=offset_0,
                        mono=True,
                    )
                    audio_data_1, orig_sr1 = librosa.load(
                        audio_fp_1,
                        sr=self.aud_params["sample_rate"],
                        offset=0.0,
                        duration=duration_1,
                        mono=True,
                    )
                    assert (
                        orig_sr0 == orig_sr1 == self.aud_params["sample_rate"]
                    ), f"Loading of audio did not go according to plan for {audio_fp}"
                    if self.aud_params["sample_rate"] == 48000:
                        audio_data_0 = int16_to_float32(float32_to_int16(audio_data_0))
                        audio_data_1 = int16_to_float32(float32_to_int16(audio_data_1))
                    audio_data_0 = torch.tensor(audio_data_0).float()
                    audio_data_1 = torch.tensor(audio_data_1).float()
                    audio_data = torch.cat((audio_data_0, audio_data_1), dim=0)
                else:
                    raise ValueError(
                        f"Need two audiofiles for this to work {audio_fp_0}, {audio_fp_1}"
                    )

        sample = self._get_audio_features(
            sample, audio_data, max_len, data_truncating, data_filling, audio_cfg
        )

        return sample

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
        video_fp, video_sec, bound_sec = self._get_video_path(sample)
        while video_fp == -1:
            print(
                f"There has been an error with sample {sample} so need to replace it with something else"
            )
            new_item = random.randint(0, len(self.metadata))
            item = new_item % len(self.metadata)
            sample = self.metadata.iloc[item]
            video_fp, video_sec, bound_sec = self._get_video_path(sample)
        caption, noun_vec, verb_vec = self._get_caption(sample)
        if self.both is True:
            # COmmenting out line below since we are not interested in the video part for now
            final = self._get_video_frames(video_fp, video_sec, bound_sec)
        else:
            final = torch.zeros((4, 3, 224, 224))

        auds = self._get_audio_dict(
            sample=sample,
            audio_ext=self.aud_params["audio_ext"],
            max_len=self.max_length,
            audio_cfg=self.aud_params,
            video_fp=video_fp,
            video_sec=video_sec,
            bound_sec=bound_sec,
            data_filling=self.aud_params["data_filling"],
            data_truncating=self.aud_params["data_truncating"],
        )
        # Scene-aware negative sampling
        if self.neg_param:
            # sample_neg = self.metadata[(self.metadata.video_uid==sample.video_uid)].sample(1).iloc[0] # variant of negative sample from same video
            sample_neg = (
                self.metadata[self.metadata.segment_id == sample.segment_id]
                .sample(1)
                .iloc[0]
            )
            video_fp_neg, video_sec_neg, bound_sec_neg = self._get_video_path(
                sample_neg
            )
            caption_neg, noun_vec_neg, verb_vec_neg = self._get_caption(sample_neg)

            if self.both is True:
                # COmmenting out line below since we are not interested in the video part for now
                final_neg = self._get_video_frames(
                    video_fp_neg, video_sec_neg, bound_sec_neg
                )
            else:
                final_neg = torch.zeros((4, 3, 224, 224))
            auds_neg = self._get_audio_dict(
                sample=sample_neg,
                audio_ext=self.aud_params["audio_ext"],
                max_len=self.max_length,
                audio_cfg=self.aud_params,
                video_fp=video_fp_neg,
                video_sec=video_sec_neg,
                bound_sec=bound_sec_neg,
                data_filling=self.aud_params["data_filling"],
                data_truncating=self.aud_params["data_truncating"],
            )

        meta_arr = {
            "raw_captions": caption,
            "paths": video_fp,
            "dataset": self.dataset_name,
        }
        if self.neg_param:
            return {
                "video": final,
                "audio": auds,
                "text": caption,
                "video_neg": final_neg,
                "audio_neg": auds_neg,
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
                "audio": auds,
                "text": caption,
                "meta": meta_arr,
                "noun_vec": noun_vec,
                "verb_vec": verb_vec,
            }

    def _replace_caps_after_person(self, sentence):
        # Regular expression to match 'a man', 'A man', 'a woman', 'A woman', 'woman', or 'man'
        # followed by a single uppercase letter and then a space, end of the sentence,
        # or any other character. The (?i) makes the match case-insensitive.
        pattern = r"(?i)((a|A)?\s?(man|woman|person)) [A-Z](?=\s|\.|$)"
        # Replace the matched pattern with the appropriate match without the capitalized letter
        return re.sub(pattern, r"\1", sentence)

    def _get_val_item(self, item):
        item = item % len(self.metadata)
        itemMCQ = self.metadata[str(item)]
        itemMCQ_gpt = self.metadata_gpt[str(item)]

        answerIndex = itemMCQ["answer"]
        sampleQuery = itemMCQ["query"]
        sampleQuery_gpt = itemMCQ_gpt["query"]
        textQuery, _, _ = self._get_caption(sampleQuery)

        # textQuery = self.orig_to_strip[textQuery]
        # textQuery = self._replace_caps_after_person(textQuery)

        textQuery_gpt = sampleQuery_gpt["clip_text"]

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
        list_keys = [
            "video_uid",
            "video_dur",
            "narration_source",
            "narration_ind",
            "narration_time",
            "clip_start",
            "clip_end",
            "clip_text",
            "tag_verb",
            "tag_noun",
            "mel_fusion",
            "longer",
            "waveform",
        ]
        audioOptions = dict((k, []) for k in list_keys)
        # audioOptions = []
        if len(sampleOptions) != 5:
            raise ValueError(
                "There is an error in the MCQ validation set. Less/more than 5 choices present"
            )
        for id, option in enumerate(sampleOptions):
            sampleOptioni = sampleOptions[option]
            video_fp, video_sec, bound_sec = self._get_video_path(sampleOptioni)
            caption, _, _ = self._get_caption(sampleOptioni)
            textOptions.append(caption)

            auds = self._get_audio_dict(
                sample=sampleOptioni,
                audio_ext=self.aud_params["audio_ext"],
                max_len=self.max_length,
                audio_cfg=self.aud_params,
                video_fp=video_fp,
                video_sec=video_sec,
                bound_sec=bound_sec,
                data_filling=self.aud_params["data_filling"],
                data_truncating=self.aud_params["data_truncating"],
            )
            if self.both is True:
                # COmmenting out line below since we are not interested in the video part for now
                imgs = self._get_video_frames(video_fp, video_sec, bound_sec)
            else:
                imgs = torch.zeros((4, 3, 224, 224))

            for list_key in list_keys:
                if list_key in auds:
                    audioOptions[list_key].append(auds[list_key])

            videoOptions[id] = imgs
        audio_keys = ["longer", "mel_fusion", "waveform"]
        for audio_key in audio_keys:
            if audio_key in auds:
                audioOptions[audio_key] = torch.stack(audioOptions[audio_key], 0)

        type_var = itemMCQ["types"]  # 1 for inter; 2 for intra
        data = {
            "video": videoOptions,
            "audio": audioOptions,
            "text": textQuery,
            "text_gpt": textQuery_gpt,
            "text_ops": textOptions,
            "correct": answerIndex,
            "type": type_var,
        }
        return data

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        if self.split == "train":
            return self._get_train_item(item)
        elif self.split in ["val", "test"]:
            return self._get_val_item(item)


if __name__ == "__main__":
    kwargs = dict(
        dataset_name="EgoClip_dataset",
        text_params={"input": "text"},
        video_params={"input_res": 224, "num_frames": 4, "loading": "lax"},
        data_dir="dataset/ego4d_256/data_chunked",
        meta_dir="dataset/ego4d_toolbox/0_metadata/egovlp",
        tsfms=init_video_transform_dict()["test"],
        reader="cv2_egoclip",
        split="val",
        neg_param=60,
    )
    dataset = EgoClip_EgoMCQ_CLAP(**kwargs)
    for i in range(100):
        item = dataset[i]
        print(item.keys())
