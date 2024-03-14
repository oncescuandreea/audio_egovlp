import os
import pickle
import random
import warnings
from contextlib import suppress
from re import sub

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

from base.base_dataset import TextAudVideoDataset
from data_loader.transforms import init_video_transform_dict
from utils import get_mel, strhhmmss_tofloats


class MultiInstanceRetrieval_CLAP(TextAudVideoDataset):
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
        suffix_train="",
        suffix="",
        relevancy_type="caption",
        val_test_split="test",
    ):
        self.suffix = suffix
        self.suffix_train = suffix_train
        self.relevancy_type = relevancy_type
        self.val_test_split = val_test_split
        print(
            f"Using relevancy_type {self.relevancy_type} and val test split: {self.val_test_split}"
        )
        super().__init__(
            dataset_name,
            text_params,
            aud_params,
            video_params,
            data_dir,
            meta_dir,
            split,
            tsfms,
            cut,
            subsample,
            sliding_window_stride,
            reader,
            neg_param,
            both,
        )
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

    def _load_metadata(self):
        print(f"Using suffix {self.suffix}")
        print(f"Using suffix_train {self.suffix_train}")
        split_files = {
            "train": f"EPIC_100_retrieval_train{self.suffix_train}.csv",
            "val": f"EPIC_100_retrieval_test{self.suffix}.csv",  # there is no test
            "test": f"EPIC_100_retrieval_test{self.suffix}.csv",
            # "val": f"EPIC_Sounds_test_epicmirstyle{self.suffix}.csv",  # there is no test
            # "test": f"EPIC_Sounds_test_epicmirstyle{self.suffix}.csv",
        }
        split_files_gpt = {
            "train": f"EPIC_100_retrieval_train{self.suffix_train}.csv",
            "val": f"EPIC_100_retrieval_test{self.suffix}_gpt.csv",  # there is no test
            "test": f"EPIC_100_retrieval_test{self.suffix}_gpt.csv",
            # "val": f"EPIC_Sounds_test_epicmirstyle{self.suffix}_gpt.csv",  # there is no test
            # "test": f"EPIC_Sounds_test_epicmirstyle{self.suffix}_gpt.csv",
        }
        split_files_sentence = {
            "train": f"EPIC_100_retrieval_train_sentence{self.suffix_train}.csv",
            "val": f"EPIC_100_retrieval_test_sentence{self.suffix}.csv",  # there is no test
            "test": f"EPIC_100_retrieval_test_sentence{self.suffix}.csv",
            # "val": f"EPIC_Sounds_test_epicmirstyle_sentence{self.suffix}.csv",
            # "test": f"EPIC_Sounds_test_epicmirstyle_sentence{self.suffix}.csv",
        }
        split_files_sentence_gpt = {
            "train": f"EPIC_100_retrieval_train_sentence{self.suffix_train}.csv",
            "val": f"EPIC_100_retrieval_test_sentence{self.suffix}_gpt.csv",  # there is no test
            "test": f"EPIC_100_retrieval_test_sentence{self.suffix}_gpt.csv",
            # "val": f"EPIC_Sounds_test_epicmirstyle_sentence{self.suffix}_gpt.csv",
            # "test": f"EPIC_Sounds_test_epicmirstyle_sentence{self.suffix}_gpt.csv",
        }
        target_split_fp = split_files[self.split]
        target_split_fp_gpt = split_files_gpt[self.split]
        self.target_split_fp = target_split_fp
        metadata = pd.read_csv(os.path.join(self.meta_dir, target_split_fp))
        metadata_gpt = pd.read_csv(os.path.join(self.meta_dir, target_split_fp_gpt))

        target_split_sentence_fp = split_files_sentence[self.split]
        target_split_sentence_fp_gpt = split_files_sentence_gpt[self.split]
        self.target_split_sentence_fp = target_split_sentence_fp
        metadata_sentence = pd.read_csv(
            os.path.join(self.meta_dir, target_split_sentence_fp)
        )
        metadata_sentence_gpt = pd.read_csv(
            os.path.join(self.meta_dir, target_split_sentence_fp_gpt)
        )

        if self.split == "train":
            self.path_relevancy = os.path.join(
                self.meta_dir,
                f"relevancy/{self.relevancy_type}_relevancy_EPIC_100_retrieval_train{self.suffix_train}.pkl",
            )
        elif self.split in ["val", "test"]:
            if "Sound" in target_split_fp:
                print("sound in")
                self.path_relevancy = os.path.join(
                    self.meta_dir,
                    f"relevancy/{self.relevancy_type}_relevancy_EPIC_Sounds_test_epicmirstyle{self.suffix}.pkl",
                )
            else:
                self.path_relevancy = os.path.join(
                    self.meta_dir,
                    f"relevancy/{self.relevancy_type}_relevancy_EPIC_100_retrieval_test{self.suffix}.pkl",
                )

        # pkl_file = open(self.path_relevancy, 'rb')
        self.relevancy = 0.1
        # self.relevancy_mat = pickle.load(pkl_file)

        self.metadata = metadata
        self.metadata_sentence = metadata_sentence
        self.metadata_gpt = metadata_gpt
        self.metadata_sentence_gpt = metadata_sentence_gpt
        self.debug = False
        if self.debug:
            warnings.warn(
                "The debug is set to True so code will be generating random values!!"
            )

    def _get_video_path(self, sample):
        rel_video_fp = sample[2]
        participant_id = rel_video_fp.split("_")[0]
        # full_video_fp = os.path.join(
        #     self.data_dir, participant_id, "rgb_frames", rel_video_fp
        # )

        full_video_fp = os.path.join(
            "/work/oncescu/data/epickitchens/",
            participant_id,
            "rgb_frames",
            rel_video_fp,
        )

        return full_video_fp, rel_video_fp

    def _get_caption(self, idx, sample, sample_gpt=None):
        # return sentence, relevancy score, idx
        if self.split == "train":
            if self.suffix_train == "":
                relevancy_location_pkl = "relevancy/caption_train_old/{}.pkl".format(
                    idx
                )
            elif self.suffix_train == "_filtered_backg":
                relevancy_location_pkl = "relevancy/caption_train/{}.pkl".format(idx)
            else:
                relevancy_location_pkl = "relevancy/caption_train{}/{}.pkl".format(
                    self.suffix_train, idx
                )
            with open(
                os.path.join(self.meta_dir, relevancy_location_pkl), "rb"
            ) as f_local:
                local_relevancy_mat = pickle.load(f_local)
            positive_list = np.where(local_relevancy_mat > self.relevancy)[0].tolist()
            if positive_list != []:
                pos = random.sample(positive_list, min(len(positive_list), 1))[0]
                if (
                    pos < len(self.metadata_sentence)
                    and pos < local_relevancy_mat.shape[0]
                ):
                    if sample_gpt is None:
                        return (
                            self.metadata_sentence.iloc[pos][1],
                            local_relevancy_mat[pos],
                            pos,
                        )
                    else:
                        return (
                            self.metadata_sentence.iloc[pos][1],
                            local_relevancy_mat[pos],
                            pos,
                            self.metadata_sentence_gpt.iloc[pos][1],
                        )
            if sample_gpt is None:
                return sample[8], 1, 0
            else:
                return sample[8], 1, 0, sample_gpt[8]

        elif self.split in ["val", "test"]:
            # narration_idx = 7 if "Sound" in self.target_split_fp else 8
            narration_idx = 8
            if sample_gpt is None:
                return sample[narration_idx], 1, -1
            else:
                return sample[narration_idx], 1, 0, sample_gpt[narration_idx]

    def _get_video_frames(self, video_fp, start_frame, stop_frame):
        video_loading = self.video_params.get("loading", "strict")
        frame_sample = "rand"
        if self.split in ["test", "val"]:
            frame_sample = "uniform"
        fix_start = None

        try:
            if os.path.exists(video_fp):
                # only supported for read_frames_decord_online
                imgs, idxs = self.video_reader(
                    video_fp,
                    start_frame,
                    stop_frame,
                    self.video_params["num_frames"],
                    frame_sample,
                    fix_start=fix_start,
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

    @staticmethod
    def _get_audio_features(
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
                    # data_fillings = ["repeatpad", "pad", "repeat"]
                    # rand_data_fil_idx = random.randint(0,2)
                    # data_filling = data_fillings[rand_data_fil_idx]
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
        start_time,
        stop_time,
        class_index_dict=None,
        data_filling="pad",
        data_truncating="rand_trunc",
        text_augment_selection=None,
    ):
        audio_fp = self.data_dir + "audio/" + sample[1] + "/" + sample[2] + ".flac"
        if os.path.isfile(audio_fp):
            duration = stop_time - start_time
            offset = start_time
            duration += self.right_sec
            if start_time - self.left_sec >= 0:
                offset -= self.left_sec
                duration += self.left_sec
            audio_data, _ = librosa.load(
                audio_fp,
                sr=self.aud_params["sample_rate"],
                mono=True,
                offset=offset,
                duration=duration,
            )
            audio_data = torch.tensor(audio_data)

        else:
            print(f"Error with {audio_fp}")
            raise ValueError("Some problem with audio content")

        sample = self._get_audio_features(
            sample,
            audio_data,
            self.max_length,
            data_truncating,
            data_filling,
            audio_cfg,
        )

        return sample

    def _text_wavcaps(self, sentence):
        # transform to lower case
        sentence = sentence.lower()

        # remove any forgotten space before punctuation and double space
        sentence = sub(r'\s([,.!?;:"](?:\s|$))', r"\1", sentence).replace("  ", " ")

        # remove punctuations
        # sentence = sub('[,.!?;:\"]', ' ', sentence).replace('  ', ' ')
        sentence = sub('[(,.!?;:|*")]', " ", sentence).replace("  ", " ")
        return sentence

    def __getitem__(self, item):
        item = item % len(self.metadata)
        random_item = item
        while random_item == item:
            random_item = random.randint(0, len(self.metadata)) % len(self.metadata)
        random_sample = self.metadata.iloc[random_item]
        sample = self.metadata.iloc[item]
        sample_gpt = self.metadata_gpt.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        caption, relation, idx, caption_gpt = self._get_caption(
            item, sample, sample_gpt
        )
        random_caption, _, _, _ = self._get_caption(item, random_sample, sample_gpt)
        start_frame, stop_frame = int(sample[6]), int(sample[7])
        if self.both is True and self.debug is False:
            # COmmenting out line below since we are not interested in the video part for now
            # print('Video features are used because both is True')
            final = self._get_video_frames(video_fp, start_frame, stop_frame)
        else:
            # Save time by not processing videos since they are not needed for trainig/prediction
            # print('Video features not used so replaced by zeros because both is False')
            final = torch.zeros((4, 3, 224, 224))

        start_time_str = sample["start_timestamp"]
        stop_time_str = sample["stop_timestamp"]
        start_time = strhhmmss_tofloats(start_time_str)
        stop_time = strhhmmss_tofloats(stop_time_str)
        if self.debug is True:
            auds = {
                "longer": torch.tensor([False]),
                "waveform": torch.rand(self.max_length),
            }
        else:
            auds = self._get_audio_dict(
                sample=sample,
                audio_ext=self.aud_params["audio_ext"],
                max_len=self.max_length,
                audio_cfg=self.aud_params,
                data_filling=self.aud_params["data_filling"],
                data_truncating=self.aud_params["data_truncating"],
                start_time=start_time,
                stop_time=stop_time,
            )

        meta_arr = {
            "raw_captions": caption,
            "paths": item,
            "dataset": self.dataset_name,
            "extra_info": f"{sample[1]}, {sample[2]}, offset:{start_time}, stop_time:{stop_time}, right_sec:{self.right_sec}, left_sec:{self.left_sec}",
        }
        # caption_aud_gpt = self._text_wavcaps(caption_gpt)
        caption_aud_gpt = caption_gpt
        data = {
            "video": final,
            "audio": auds,
            "text": caption,
            # "text": random_caption,
            "text_gpt": caption_aud_gpt,
            "meta": meta_arr,
            "relation": relation,
            "item_v": item,
            "item_t": idx,
            # "train_sentences": self.train_sentences,
        }
        return data


if __name__ == "__main__":
    kwargs = dict(
        dataset_name="MultiInstanceRetrieval_CLAP",
        text_params={"input": "text"},
        video_params={"input_res": 224, "num_frames": 4, "loading": "lax"},
        data_dir="dataset/epic-kitchens/epic-kitchens-rgb-frames",
        meta_dir="dataset/epic-kitchens/epic-kitchens-100-annotations-master/retrieval_annotations",
        tsfms=init_video_transform_dict()["test"],
        reader="cv2_epic",
        split="train",
    )
    dataset = MultiInstanceRetrieval_CLAP(**kwargs)
    for i in range(100):
        item = dataset[i]
        print(item.keys())
