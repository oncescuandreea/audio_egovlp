import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool, Value

import tqdm

# folder_path = '/ego4d/data'
folder_path = "./dataset/ego4d/"
output_path = "./dataset/ego4d_256/"


def videos_resize(videoinfos):
    global count

    videoid, videoname = videoinfos

    if os.path.exists(os.path.join(output_path, videoname)):
        print(f"{videoname} is resized.")
        return

    inname = folder_path + "/" + videoname
    outname = output_path + "/" + videoname

    cmd = 'ffmpeg -y -i {} -filter:v scale="trunc(oh*a/2)*2:256" -c:a copy {}'.format(
        inname, outname
    )
    try:
        subprocess.call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Command '{e.cmd}' returned non-zero exit status '{e.returncode}'.")
        print(f"Command output: {e.output}")
        print(f"Command stderr: {e.stderr}")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resizing video files")
    parser.add_argument(
        "--nproc",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        choices=["all", "mcq"],
    )
    args = parser.parse_args()
    file_list = []
    if os.path.exists("dataset/ego4d_256") is False:
        os.makedirs("dataset/ego4d_256")
        existent_mp4s = []
    else:
        existent_mp4s = os.listdir(output_path)
        existent_mp4s = [item for item in existent_mp4s if item.endswith(".mp4")]
    if args.subset == "all":
        mp4_list = [item for item in os.listdir(folder_path) if item.endswith(".mp4")]
    else:
        print("got here correctly")
        with open(
            "dataset/egomcq_aud_full_filtered_query_and_answer_filter_cliptextfull_silence.json",
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

    for id, video in enumerate(tqdm.tqdm(mp4_list)):
        if video in existent_mp4s:
            print("already extracted this")
        else:
            file_list.append([id, video])

    pool = Pool(args.nproc)
    pool.map(videos_resize, tuple(file_list))
    pool.close()
    pool.join()
