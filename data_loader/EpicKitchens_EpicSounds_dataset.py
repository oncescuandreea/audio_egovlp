import os
import random
import warnings

import pandas as pd
import torch

from data_loader.EpicKitchens_MIR_dataset_CLAP import MultiInstanceRetrieval_CLAP
from data_loader.transforms import init_video_transform_dict
from utils import strhhmmss_tofloats


class EpicKitchens_EpicSounds(MultiInstanceRetrieval_CLAP):
    def _load_metadata(self):
        print(f"Using suffix {self.suffix}")
        print(f"Using suffix_train {self.suffix_train}")
        # val_test_split = "test"
        print(f"Using val_test_split {self.val_test_split}")
        split_files = {
            "train": f"EPIC_100_retrieval_train{self.suffix_train}.csv",
            "val": f"EPIC_Sounds_{self.val_test_split}_epicmirstyle_mainclasses{self.suffix}.csv",  # there is no test
            "test": f"EPIC_Sounds_{self.val_test_split}_epicmirstyle_mainclasses{self.suffix}.csv",
            # "val": f"EPIC_Sounds_{self.val_test_split}_epicmirstyle_mainclasses{self.suffix}_gpt.csv",  # there is no test
            # "test": f"EPIC_Sounds_{self.val_test_split}_epicmirstyle_mainclasses{self.suffix}_gpt.csv",
        }
        split_files_gpt = {
            "train": f"EPIC_100_retrieval_train{self.suffix_train}.csv",
            "val": f"EPIC_Sounds_{self.val_test_split}_epicmirstyle_mainclasses{self.suffix}_gpt.csv",  # there is no test
            "test": f"EPIC_Sounds_{self.val_test_split}_epicmirstyle_mainclasses{self.suffix}_gpt.csv",
        }
        split_files_sentence = {
            "train": f"EPIC_100_retrieval_train_sentence{self.suffix_train}.csv",
            "val": f"EPIC_Sounds_{self.val_test_split}_epicmirstyle_sentence_mainclasses{self.suffix}.csv",
            "test": f"EPIC_Sounds_{self.val_test_split}_epicmirstyle_sentence_mainclasses{self.suffix}.csv",
            # "val": f"EPIC_Sounds_{self.val_test_split}_epicmirstyle_sentence_mainclasses{self.suffix}_gpt.csv",
            # "test": f"EPIC_Sounds_{self.val_test_split}_epicmirstyle_sentence_mainclasses{self.suffix}_gpt.csv",
        }
        split_files_sentence_gpt = {
            "train": f"EPIC_100_retrieval_train_sentence{self.suffix_train}.csv",
            "val": f"EPIC_Sounds_{self.val_test_split}_epicmirstyle_sentence_mainclasses{self.suffix}_gpt.csv",
            "test": f"EPIC_Sounds_{self.val_test_split}_epicmirstyle_sentence_mainclasses{self.suffix}_gpt.csv",
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
            self.path_relevancy = os.path.join(
                self.meta_dir,
                f"relevancy/{self.relevancy_type}_relevancy_EPIC_Sounds_{self.val_test_split}_epicmirstyle_mainclasses{self.suffix}.pkl",
                # f"relevancy/{self.relevancy_type}_relevancy_EPIC_Sounds_{val_test_split}_epicmirstyle_mainclasses{self.suffix}_vid_gpt.pkl",
            )

        self.relevancy = 0.1

        self.metadata = metadata
        self.metadata_sentence = metadata_sentence
        self.metadata_gpt = metadata_gpt
        self.metadata_sentence_gpt = metadata_sentence_gpt
        self.debug = False
        if self.debug:
            warnings.warn(
                "The debug is set to True so code will be generating random values!!"
            )

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
            print("Loading true frames")
            # COmmenting out line below since we are not interested in the video part for now
            # print('Video features are used because both is True')
            final = self._get_video_frames(video_fp, start_frame, stop_frame)
        else:
            print("Loading empty frames")
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
        dataset_name="MultiInstanceRetrieval",
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
