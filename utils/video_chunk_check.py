import argparse
import json
import os
import shutil
import subprocess
import sys
from csv import reader, writer
from multiprocessing import Pool
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

output_dir_epic = "/datasets/EpicKitchens-100/"
output_aud_epic = "./data/epic-kitchens-stereo/"

video_dir = "/scratch/shared/beegfs/shared-datasets/EGO4D/ego4d_data/v1/full_scale"
video_dir_256 = "./dataset/ego4d_256/"
output_dir = "./dataset/ego4d_chunked/"

output_aud_dir = "./dataset/ego4d_chunked_audio/"


dur_limit = 600


def segments2aud(infos):
    cmd = "ffmpeg -y -i {} -f flac -ar 48000 -ac 2 -async 1 -vn {}".format(
        infos[0], infos[1]
    )
    # print(cmd)
    subprocess.call(cmd, shell=True)


def aud2dur(infos):
    cmd = "ffprobe -i {} -show_entries format=duration".format(infos[0])
    duration = subprocess.check_output(cmd, shell=True)
    duration = duration.decode("utf-8")
    duration = duration.split("\n")[1].strip("duration=")
    return duration


def video2segments(infos):
    global count
    index, uid, dur = infos[0], infos[1], infos[2]
    input_path = os.path.join(video_dir, uid + ".mp4")

    output_uid_dir = os.path.join(output_dir, uid)
    if not os.path.exists(output_uid_dir):
        os.makedirs(output_uid_dir)

    # if index % num_partition != partition:
    #     return

    assert os.path.exists(input_path)

    cap = cv2.VideoCapture(input_path)
    rate = cap.get(5)
    frame_num = cap.get(7)
    duration = frame_num / rate

    num_seg = duration // dur_limit

    no_chunks_extracted = len(os.listdir(output_uid_dir))
    if no_chunks_extracted != int(num_seg) + 1:
        try:
            os.remove(os.path.join(video_dir_256, uid + ".mp4"))
            print(f"removed {uid}")
            print(uid)
        except Exception as e:
            print(e)
    return


def video2segments_ch(infos):
    global count
    index, uid, dur = infos[0], infos[1], infos[2]
    input_path = os.path.join(video_dir, uid + ".mp4")
    input_path_aud = os.path.join(output_aud_dir, uid + ".flac")

    output_uid_dir = os.path.join(output_dir, uid)
    if not os.path.exists(output_uid_dir):
        os.makedirs(output_uid_dir)

    assert os.path.exists(input_path)

    cap = cv2.VideoCapture(input_path)
    rate = cap.get(5)
    frame_num = cap.get(7)
    duration = frame_num / rate

    num_seg = duration // dur_limit

    no_chunks_extracted = len(os.listdir(output_uid_dir))
    no_aud_chunks_extracted = len(os.listdir(input_path_aud))
    if no_chunks_extracted != no_aud_chunks_extracted:
        print(f"Diff no of chunks in {uid} aud and vid")
    if no_chunks_extracted != int(num_seg) + 1:
        try:
            print(f"No extracted vid segments makes no sense {uid}")
        except Exception as e:
            print(e)
    return


def segments2aud_ch(infos):
    video_chunk_path = infos[0]
    audio_chunk_path = infos[1]
    folder, file = audio_chunk_path.split("/")[-2:]
    file = file.rsplit(".", 1)[0]
    if (
        os.path.exists(audio_chunk_path) is False
        and os.path.exists(video_chunk_path) is True
    ):
        with open(
            f"txt_files/{folder}_{file}_mis.txt",
            "w",
        ) as f:
            f.write(audio_chunk_path)
        print(f"Audio file {audio_chunk_path} missing")
    elif (
        os.path.exists(audio_chunk_path) is True
        and os.path.exists(video_chunk_path) is True
    ):
        # return
        aud_dur = float(aud2dur([audio_chunk_path]))
        vid_dur = float(aud2dur([video_chunk_path]))
        if vid_dur - 0.5 <= aud_dur <= vid_dur + 0.5 is False:
            print(f"Audio file {audio_chunk_path} wrongly extracted")
            with open(
                f"txt_files/{folder}_{file}_incomplete.txt",
                "w",
            ) as f:
                f.write(audio_chunk_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="audio_chunks",
        choices=["video_chunks", "audio_chunks", "duration"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="egovlp",
        choices=["egovlp", "epickitchens"],
    )

    args = parser.parse_args()
    if args.task == "video_chunks":
        with open("./dataset/manifest.csv", "r") as csv:
            csv_reader = list(reader(csv))[1:]

        with open("./dataset/egomcq_aud_full_filtered.json", "r") as f:
            egocmq = json.load(f)
        mp4_list = []
        for entry_idx, entry in egocmq.items():
            query_vid = entry["query"]["video_uid"]
            if f"{query_vid}.mp4" not in mp4_list:
                mp4_list.append(f"{query_vid}.mp4")
            for i in range(0, 5):
                answer_vid = entry["choices"][str(i)]["video_uid"]
                if f"{answer_vid}.mp4" not in mp4_list:
                    mp4_list.append(f"{answer_vid}.mp4")
        downloaded = mp4_list

        already_chunked = os.listdir(output_dir)
        uid_list = []
        infos_list = []
        num_valid = 0
        for i_item, item in enumerate(csv_reader):
            uid, dur = item[0], float(item[2])
            existed = uid + ".mp4" in downloaded

            if not existed:
                continue

            # if uid not in already_chunked:
            uid_list.append(uid)
            infos_list.append([num_valid, uid, dur])
            num_valid += 1

        pool = Pool(10)
        pool.map(video2segments, tuple(infos_list))
        pool.close()
        pool.join()
    elif args.task == "audio_chunks":
        if args.dataset == "egovlp":
            folders = os.listdir(output_dir)
            infos_list = []
            for folder in tqdm(folders):
                files = os.listdir(Path(output_dir) / folder)
                (Path(output_aud_dir) / folder).mkdir(parents=True, exist_ok=True)
                for file_v in files:
                    file_a = file_v.rsplit(".mp4", 1)[0]
                    output_flac_path = os.path.join(
                        output_aud_dir, folder, file_a + ".flac"
                    )
                    input_mp4_path = os.path.join(output_dir, folder, file_v)
                    infos_list.append([input_mp4_path, output_flac_path])
        elif args.dataset == "epickitchens":
            Path(output_aud_epic).mkdir(parents=True, exist_ok=True)
            folders = os.listdir(output_dir_epic)
            infos_list = []
            for folder in tqdm(folders):
                if "P" in folder:
                    files = os.listdir(Path(output_dir_epic) / folder / "videos")
                    (Path(output_aud_epic) / folder).mkdir(parents=True, exist_ok=True)
                    for file_v in files:
                        file_a = file_v.rsplit(".MP4", 1)[0]
                        output_flac_path = os.path.join(
                            output_aud_epic, folder, file_a + ".flac"
                        )
                        input_mp4_path = os.path.join(
                            output_dir_epic, folder, "videos", file_v
                        )
                        infos_list.append([input_mp4_path, output_flac_path])

        pool = Pool(30)
        pool.map(segments2aud_ch, tuple(infos_list))
        pool.close()
        pool.join()
    else:
        folders = os.listdir(output_aud_dir)
        infos_list = []
        for folder in tqdm(folders):
            files = os.listdir(Path(output_aud_dir) / folder)
            for file_a in files:
                file_t = file_a.strip(".flac")
                input_flac_path = os.path.join(output_aud_dir, folder, file_a)
                infos_list.append([input_flac_path])

        pool = Pool(32)
        pool.map(aud2dur, tuple(infos_list))
