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
output_aud_epic = "./data/epic-kitchens/"

video_dir = "./dataset/ego4d_256/"
output_dir = "./dataset/ego4d_chunked/"

output_aud_dir = "./dataset/ego4d_chunked_audio/"


dur_limit = 600


def segments2aud(infos):
    cmd = "ffmpeg -y -i {} -f flac -ar 48000 -ac 1 -async 1 -vn {}".format(
        infos[0], infos[1]
    )
    print(cmd)
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

    assert os.path.exists(input_path), f"input path {input_path} seems to not exist"

    cap = cv2.VideoCapture(input_path)
    rate = cap.get(5)
    frame_num = cap.get(7)
    duration = frame_num / rate

    if duration <= dur_limit:
        output_mp4_path = os.path.join(output_uid_dir, "0.mp4")
        if os.path.exists(output_mp4_path) is False:
            shutil.copyfile(input_path, output_mp4_path)
        else:
            print(f"Already extracted {output_mp4_path}")
        return

    num_seg = duration // dur_limit

    s1_time = 0
    s2_time = dur_limit
    num_finished = 0
    while num_finished <= num_seg:
        output_mp4_path = os.path.join(output_uid_dir, str(num_finished) + ".mp4")
        if os.path.exists(output_mp4_path) is False:
            cmd = "ffmpeg -y -i {} -ss {} -to {} -async 1 {}".format(
                input_path, s1_time, s2_time, output_mp4_path
            )
            print(cmd)
            subprocess.call(cmd, shell=True)
        else:
            print(f"Already extracted {output_mp4_path}")

        # Update for next steps
        s1_time = s2_time
        s2_time = min(s1_time + dur_limit, duration)
        num_finished += 1
    return


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
    parser.add_argument(
        "--nproc",
        type=int,
        default=10,
    )
    args = parser.parse_args()
    if args.task == "video_chunks":
        if os.path.exists("dataset/ego4d_chunked") is False:
            os.makedirs("dataset/ego4d_chunked")
        with open("./dataset/manifest.csv", "r") as csv:
            csv_reader = list(reader(csv))[1:]

        # downloaded = os.listdir("./dataset/ego4d")
        with open(
            "./dataset/egomcq_aud_full_filtered_query_and_answer_filter_cliptextfull_silence.json",
            "r",
        ) as f:
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

            uid_list.append(uid)
            infos_list.append([num_valid, uid, dur])
            num_valid += 1

        pool = Pool(args.nproc)
        pool.map(video2segments, tuple(infos_list))
        pool.close()
        pool.join()
    elif args.task == "audio_chunks":
        if args.dataset == "egovlp":
            if os.path.exists(output_aud_dir) is False:
                os.makedirs(output_aud_dir)
            ### This is added to only extract audios for videos where audio exists
            with open(
                "./dataset/egomcq_aud_full_filtered_query_and_answer_filter_cliptextfull_silence.json",
                "r",
            ) as f:
                egocmq = json.load(f)
            audio_folders_list = []
            for entry_idx, entry in egocmq.items():
                query_vid = entry["query"]["video_uid"]
                if query_vid not in audio_folders_list:
                    audio_folders_list.append(query_vid)
                for i in range(0, 5):
                    answer_vid = entry["choices"][str(i)]["video_uid"]
                    if answer_vid not in audio_folders_list:
                        audio_folders_list.append(answer_vid)
            ################ done

            # folders = os.listdir(output_dir)
            folders = audio_folders_list
            infos_list = []
            for folder in tqdm(folders):
                files = os.listdir(Path(output_dir) / folder)
                (Path(output_aud_dir) / folder).mkdir(parents=True, exist_ok=True)
                for file_v in files:
                    file_a = file_v.rsplit(".mp4", 1)[0]
                    output_flac_path = os.path.join(
                        output_aud_dir, folder, file_a + ".flac"
                    )
                    if os.path.exists(output_flac_path):
                        continue
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
                        # if os.path.exists(output_flac_path):
                        #     continue
                        input_mp4_path = os.path.join(
                            output_dir_epic, folder, "videos", file_v
                        )
                        infos_list.append([input_mp4_path, output_flac_path])

        pool = Pool(args.nproc)
        pool.map(segments2aud, tuple(infos_list))
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
