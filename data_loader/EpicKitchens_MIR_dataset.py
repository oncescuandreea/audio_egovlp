import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from base.base_dataset import TextVideoDataset
from data_loader.transforms import init_transform_dict, init_video_transform_dict


class MultiInstanceRetrieval(TextVideoDataset):
    def __init__(
        self,
        dataset_name,
        text_params,
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
        suffix_train="",
        suffix="",
        relevancy_type="caption",
    ):
        # suffix = ""
        # suffix = "_gptfiltered_moderate"
        # suffix_train = ""
        self.suffix = suffix
        self.suffix_train = suffix_train
        self.relevancy_type = relevancy_type
        print(f"Using relevancy_type {self.relevancy_type}.")
        super().__init__(
            dataset_name,
            text_params,
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
        )
        # self.cumulative_file_counts, self.total_files = self.build_file_counts()

    def _load_metadata(self):
        print(f"Using suffix {self.suffix}")
        print(f"Using suffix_train {self.suffix_train}")
        split_files = {
            "train": f"EPIC_100_retrieval_train{self.suffix_train}.csv",
            # 'val': 'EPIC_100_retrieval_test.csv',            # there is no test
            # 'test': 'EPIC_100_retrieval_test.csv'
            "val": f"EPIC_100_retrieval_test{self.suffix}.csv",  # there is no test
            "test": f"EPIC_100_retrieval_test{self.suffix}.csv",
        }
        split_files_sentence = {
            "train": f"EPIC_100_retrieval_train_sentence{self.suffix_train}.csv",
            # 'val': 'EPIC_100_retrieval_test_sentence.csv',  # there is no test
            # 'test': 'EPIC_100_retrieval_test_sentence.csv'
            "val": f"EPIC_100_retrieval_test_sentence{self.suffix}.csv",  # there is no test
            "test": f"EPIC_100_retrieval_test_sentence{self.suffix}.csv",
        }
        target_split_fp = split_files[self.split]
        self.target_split_fp = target_split_fp
        metadata = pd.read_csv(os.path.join(self.meta_dir, target_split_fp))

        target_split_sentence_fp = split_files_sentence[self.split]
        self.target_split_sentence_fp = target_split_sentence_fp
        metadata_sentence = pd.read_csv(
            os.path.join(self.meta_dir, target_split_sentence_fp)
        )

        if self.split == "train":
            self.path_relevancy = os.path.join(
                self.meta_dir,
                f"relevancy/{self.relevancy_type}_relevancy_EPIC_100_retrieval_train{self.suffix_train}.pkl",
            )
        elif self.split in ["val", "test"]:
            self.path_relevancy = os.path.join(
                self.meta_dir,
                f"relevancy/{self.relevancy_type}_relevancy_EPIC_100_retrieval_test{self.suffix}.pkl",
            )

        pkl_file = open(self.path_relevancy, "rb")
        self.relevancy = 0.1
        self.relevancy_mat = pickle.load(pkl_file)

        self.metadata = metadata
        self.metadata_sentence = metadata_sentence

    def _get_video_path(self, sample):
        rel_video_fp = sample[2]
        participant_id = rel_video_fp.split("_")[0]
        full_video_fp = os.path.join(
            self.data_dir, participant_id, "rgb_frames", rel_video_fp
        )
        # full_video_fp = os.path.join(self.data_dir, rel_video_fp)

        return full_video_fp, rel_video_fp

    def _get_caption(self, idx, sample):
        # return sentence, relevancy score, idx
        if self.split == "train":
            positive_list = np.where(self.relevancy_mat[idx] > self.relevancy)[
                0
            ].tolist()
            if positive_list != []:
                pos = random.sample(positive_list, min(len(positive_list), 1))[0]
                if (
                    pos < len(self.metadata_sentence)
                    and pos < self.relevancy_mat.shape[1]
                ):
                    return (
                        self.metadata_sentence.iloc[pos][1],
                        self.relevancy_mat[idx][pos],
                        pos,
                    )
            return sample[8], 1, 0

        elif self.split in ["val", "test"]:
            return sample[8], 1, -1

    def __getitem__(self, item):
        item = item % len(self.metadata)
        random_item = item
        while random_item == item:
            random_item = random.randint(0, len(self.metadata)) % len(self.metadata)
        random_sample = self.metadata.iloc[random_item]
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        caption, relation, idx = self._get_caption(item, sample)
        random_caption, _, _ = self._get_caption(item, random_sample)

        video_loading = self.video_params.get("loading", "strict")
        start_frame, stop_frame = int(sample[6]), int(sample[7])

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
                imgs = transforms.ToTensor()(imgs).unsqueeze(0)

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

        meta_arr = {
            "raw_captions": caption,
            "paths": item,
            "dataset": self.dataset_name,
            "extra_info": f"{sample[2]}, start_frame:{start_frame}, stop_frame:{stop_frame}",
        }
        data = {
            "video": final,
            # "text": caption,
            "text": random_caption,
            "meta": meta_arr,
            "relation": relation,
            "item_v": item,
            "item_t": idx,
        }
        return data


if __name__ == "__main__":
    kwargs = dict(
        dataset_name="MultiInstanceRetrieval",
        text_params={"input": "text"},
        video_params={"input_res": 224, "num_frames": 4, "loading": "lax"},
        data_dir="dataset/epic-kitchens/epic-kitchens-rgb-frames",
        meta_dir="dataset/epic-kitchens/epic-kitchens-100-annotations-master/retrieval_annotations",
        tsfms=init_video_transform_dict()["test"],
        reader="cv2_epic",
        split="train",
    )
    dataset = MultiInstanceRetrieval(**kwargs)
    for i in range(100):
        item = dataset[i]
        print(item.keys())
