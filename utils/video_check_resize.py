### This file was used to check if the resized videos are valid or not. If not, they are deleted.
import os
import subprocess
from multiprocessing import Pool


def validate_file(file_path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    try:
        subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        return None
    except subprocess.CalledProcessError:
        print(f"Deleting invalid file: {file_path}")
        os.remove(file_path)
        return file_path


def validate_files_in_folder(folder_path):
    pool = Pool(40)
    file_paths = [
        os.path.join(folder_path, file_name)
        for file_name in os.listdir(folder_path)
        if file_name.endswith(".mp4")
    ]
    deleted_files = pool.map(validate_file, file_paths)
    pool.close()
    pool.join()

    # Filter out the None values and print deleted files
    deleted_files = list(filter(None, deleted_files))
    print("Deleted files:", deleted_files)


validate_files_in_folder("dataset/ego4d_256")
# folder_path = "/ego4d/data"
# output_path = "/ego4d_256/data"
