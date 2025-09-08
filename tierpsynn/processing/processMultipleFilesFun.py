# -*- coding: utf-8 -*-
"""
Created on 12 June 2024

@author: Weheliye
"""
# %%

import argparse
import gc
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import h5py
import tqdm

import tierpsynn as tnn
from tierpsy.processing.helper import find_valid_files


# %%
def _testFileExists(fname: os.PathLike):
    fname = Path(fname).absolute()
    if not Path(fname).exists():
        raise FileNotFoundError(f"{fname} does not exist.")
    return fname


def check_string(input_vid):
    if "RawVideos" not in input_vid:
        raise ValueError("The string does not contain 'RawVideos'.")
    return "The string contains 'RawVideos'."


def execute_command(command):
    os.system(command)
    gc.collect()


def create_cmd(video_file, params_well, apply_spline, action):
    return f"tierpsynn_process  --input_vid={video_file} --params_well={params_well} --apply_spline={apply_spline} --{action}=True"


def str_or_int(value):
    try:
        return int(value)
    except ValueError:
        return str(value)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "t", "yes", "y", "1"}:
        return True
    elif value.lower() in {"false", "f", "no", "n", "0"}:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# Function to check if the HDF5 file is empty or contains corrupt datasets
def is_hdf5_empty(file_path):
    try:
        with h5py.File(file_path, "r") as f:
            # Check if there are at least 5 keys
            if len(f.keys()) < 4:
                return True

            # Now, check if any key (group or dataset) is empty or corrupt
            for key in f.keys():
                item = f[key]
                # Check if the item is a group or a dataset
                if isinstance(item, h5py.Group):
                    # If it's a group, check if the group has any datasets inside
                    if len(item.keys()) == 0:
                        return True
                elif isinstance(item, h5py.Dataset):
                    # If it's a dataset, check if it contains any elements and is not corrupt
                    try:
                        # Access the dataset data to ensure it's not corrupt
                        if item.size == 0:
                            return True
                        _ = item[
                            ()
                        ]  # Try accessing the dataset to catch any corruption
                    except (OSError, IOError):
                        # Dataset is corrupt
                        return True
            return False  # File is not empty or corrupt
    except OSError:
        # Handle the case where the file itself is corrupt or can't be opened
        return True


def _return_videos(valid_files):
    valid_video_files = []
    corrupt_video_files = []
    for vf in tqdm.tqdm(
        valid_files, total=len(valid_files), desc="Checking if videos are corrupt"
    ):
        #
        with tnn.selectVideoReader(vf) as video_reader:
            if video_reader.height != 0:
                valid_video_files.append(vf)
            else:
                corrupt_video_files.append(vf)

    return valid_video_files, corrupt_video_files


# Function to check both skeletonNN.hdf5 and metadata_featuresN.hdf5 for a video
def check_video_files(vf):
    skeleton_path = (
        Path(vf.replace("RawVideos", "Results_NN")).parent / "skeletonNN.hdf5"
    )
    metadata_path = (
        Path(vf.replace("RawVideos", "Results_NN")).parent / "metadata_featuresN.hdf5"
    )

    skeleton_empty = is_hdf5_empty(skeleton_path)
    metadata_empty = is_hdf5_empty(metadata_path)

    return vf, skeleton_empty, metadata_empty


# Function to process the video files in parallel using multiprocessing
def parallel_check_hdf5(valid_video_files, params_well, apply_spline, num_processors=4):
    total_files = len(valid_video_files)
    with ProcessPoolExecutor(max_workers=num_processors) as executor:
        futures = {
            executor.submit(check_video_files, vf): vf for vf in valid_video_files
        }

        cmd_detect_track = []
        cmd_features = []

        # Collect results as they complete

        for future in tqdm.tqdm(
            as_completed(futures),
            total=total_files,
            desc="This is to check if any videos are processed",
        ):
            vf, skeleton_empty, metadata_empty = future.result()

            # If skeletonNN.hdf5 is empty, add to cmd_detect_track
            if skeleton_empty:
                cmd_detect_track.append(
                    create_cmd(vf, params_well, apply_spline, "track_detect")
                )

            # If metadata_featuresN.hdf5 is empty, add to cmd_features
            if metadata_empty:
                cmd_features.append(
                    create_cmd(vf, params_well, apply_spline, "get_features")
                )

    return cmd_detect_track, cmd_features


# %%


def main(
    input_vid,
    params_well,
    apply_spline,
    process_type,
    track_detect=True,
    post_process=True,
    file_extension="*.yaml",
):
    """_summary_

    Args:
        input_vid (list of raw video paths): Path of the RawVideos
        params_well ( string or number): parameters file or if its Hydra must input 6,24,96
        apply_spline (boolean): Apply spline if you have high fps
        process_type (string): HPC or local. Use local for default.
        track_detect (bool, optional): track and detect. Defaults to False.
        post_process (bool, optional): feature extraction. Defaults to True.
        file_extension (string, optional): file extension. Defaults to *.yaml.
    """
    video_dir_root = _testFileExists(input_vid)

    check_string(input_vid)

    valid_files = find_valid_files(
        root_dir=video_dir_root, pattern_include=[file_extension]
    )
    log_dir = video_dir_root.with_name("log")

    Path(log_dir).mkdir(exist_ok=True, parents=True)

    valid_video_files, corrupt_video_files = _return_videos(valid_files)

    # store the corrupt videos
    with open(Path(log_dir) / "corrupt_videos.txt", "w") as log_file:
        for corrupt_file in corrupt_video_files:
            log_file.write(f"{corrupt_file}\n")

    cmd_detect_track, cmd_features = parallel_check_hdf5(
        valid_video_files, params_well, apply_spline, num_processors=8
    )

    if process_type == "HPC":
        today_str = str(date.today())
        with (
            open(Path(log_dir) / f"detect_track_{today_str}.txt", "w") as dt,
            open(Path(log_dir) / f"features_{today_str}.txt", "w") as fet,
        ):
            dt.write("\n".join(cmd_detect_track))
            fet.write("\n".join(cmd_features))
    else:
        if track_detect:
            # print("Weheliye")
            for execute_dst_cmd in tqdm.tqdm(cmd_detect_track):
                os.system(execute_dst_cmd)
        if post_process:
            # with mp.Pool(processes=10) as pool:
            #     pool.starmap(execute_command, cmd_features)

            for i, results in enumerate(cmd_features):
                execute_command(results)
                print(f"finished processing file {i}")


def process_main_file():
    parser = argparse.ArgumentParser(description="Create a text file for executing.")
    parser.add_argument("--input_vid", type=str, help="Path to the input video file")
    parser.add_argument(
        "--params_well",
        type=str_or_int,
        required=True,
        help="Path to the params well file. For Hydra type the well number (i.e. 6 for 6 well plate)",
    )
    parser.add_argument(
        "--apply_spline",
        type=str2bool,
        default=False,
        help="Apply spline. Either False or True",
    )
    parser.add_argument(
        "--process_type",
        type=str,
        default="local",
        help="The type of process. Either HPC or local.",
    )
    parser.add_argument(
        "--track_detect",
        type=str2bool,
        default=True,
        help="If you are using local process you might want to only process",
    )
    parser.add_argument(
        "--post_process",
        type=str2bool,
        default=False,
        help="If you are using local process you might want to only post process",
    )
    parser.add_argument(
        "--file_extension",
        type=str,
        default="*.yaml",
        help="Please define the extension of your file if its *.yaml, *.hdf5 or *.mp4",
    )

    args = parser.parse_args()
    # print(args.track_detect, args.apply_spline)

    main(
        # args
        args.input_vid,
        args.params_well,
        args.apply_spline,
        args.process_type,
        args.track_detect,
        args.post_process,
        args.file_extension,
    )


if __name__ == "__main__":
    process_main_file()
