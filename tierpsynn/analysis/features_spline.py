# %%
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tables as tb
import tqdm

import tierpsynn as tnn
import tierpsynn.extras.cython_files.spline_cython as sp
from deeptangle.predict import Predictions
from tierpsy.analysis.ske_orient.checkHeadOrientation import isWormHTSwitched
from tierpsy.features.tierpsy_features import features
from tierpsy.helper.misc import TABLE_FILTERS
from tierpsy.helper.params import set_unit_conversions

# %%%


"""
Correct skeleton orientation
"""


def orientWorm(skeleton, prev_skeleton):
    """_summary_

    Args:
        skeleton (array): List of skeletons at time t
        prev_skeleton (array of skeleton): array of skeleton at time t-1

    Returns:
        array :  returns an a rotated skeleton
    """
    if skeleton.size == 0:
        return skeleton
    if prev_skeleton.size > 0:
        dist2prev_head = np.sum((skeleton - prev_skeleton) ** 2)
        dist2prev_tail = np.sum((skeleton - prev_skeleton[::-1, :]) ** 2)
        if dist2prev_head > dist2prev_tail:
            # the skeleton is switched
            skeleton = skeleton[::-1, :]
    return skeleton


def correct_skeleton_orient(skeleton):
    skeleton_orientworm = np.zeros(skeleton.shape)
    skeleton_orientworm[0, :] = skeleton[0, :]
    for i in range(1, skeleton.shape[0]):
        skeleton_orientworm[i, :] = orientWorm(
            skeleton[i, :], skeleton_orientworm[i - 1, :]
        )
    return skeleton_orientworm


def _return_fill_gap(frame_window: np.array, skeleton: np.array):
    new_frame_window = np.arange(frame_window[0], frame_window[-1] + 1)
    new_skeletons = np.full(
        (new_frame_window.shape[0], skeleton.shape[1], skeleton.shape[2]), np.nan
    )

    # Fill in the skeletons
    for i, frame in enumerate(frame_window):
        index = frame - frame_window[0]
        new_skeletons[index] = skeleton[i]
    return new_skeletons, new_frame_window


def linear_interpolate(skeletons, frame_indices, target_length=10):
    """Interpolate skeleton data to ensure each segment has the target length."""
    interpolated_skeleton = np.zeros((target_length, skeletons.shape[1], 2))
    interpolated_frames = np.linspace(
        frame_indices[0], frame_indices[-1], target_length
    ).astype(int)

    for joint_idx in range(skeletons.shape[1]):
        for coord_idx in range(2):
            interpolated_skeleton[:, joint_idx, coord_idx] = np.interp(
                interpolated_frames, frame_indices, skeletons[:, joint_idx, coord_idx]
            )

    return interpolated_skeleton, interpolated_frames


def _min_segment_length(step_size: int):
    return {1: 10, 2: 6, 3: 5, 4: 4}.get(step_size, 3)


def segment_data_continuous_with_constraints(
    skeletons, frame_indices, step_size, max_gap=10, max_segment_length=10
):
    """Segment skeleton data using a sliding window approach with constraints on gap and segment length."""
    segments, frame_segments = [], []
    current_segment, current_frames = [skeletons[0]], [frame_indices[0]]
    min_segment_length = _min_segment_length(
        step_size
    )  # 3 if step_size >= 4 else max_segment_length

    for start_idx in range(1, len(frame_indices)):
        if (
            frame_indices[start_idx] - frame_indices[start_idx - 1] > max_gap
            or len(current_segment) >= max_segment_length
        ):
            if len(current_frames) >= 2:
                segments.append(np.array(current_segment))
                frame_segments.append(np.array(current_frames))

            current_segment, current_frames = [], []

            if (frame_indices[start_idx] - frame_indices[start_idx - 1]) <= max_gap:
                current_segment.append(skeletons[start_idx - 1])
                current_frames.append(frame_indices[start_idx - 1])

        current_segment.append(skeletons[start_idx])
        current_frames.append(frame_indices[start_idx])

    # Add the last batch to segments
    if len(current_frames) >= min_segment_length:
        segments.append(np.array(current_segment))
        frame_segments.append(np.array(current_frames))

    # Interpolate segments to ensure length is 10
    interpolated_segments, interpolated_frame_segments = [], []
    for segment, frames in zip(segments, frame_segments):
        if min_segment_length < len(segment) < max_segment_length:
            interpolated_segment, interpolated_frames = linear_interpolate(
                segment, frames
            )
            interpolated_segments.append(interpolated_segment)
            interpolated_frame_segments.append(interpolated_frames)
        else:
            interpolated_segments.append(segment)
            interpolated_frame_segments.append(frames)

    return interpolated_segments, interpolated_frame_segments


def _return_spline_skeleton(skeleton, frame_window, Ma, Mb=10):
    # Number of spline parameters along the time axis (this will have to be tuned) - must be < number of time steps-4
    spline_sheet = sp.SplineSheet((Ma, Mb))

    frame_max = frame_window.ptp()
    point_max = skeleton.shape[1]

    worm_i = (
        frame_window - frame_window.min()
    )  # np.arange(len(skeleton))  # Assuming worm_i is a sequence of frame indices
    sample_points = np.array(
        [(ell[0], ell[1], t) for el, t in zip(skeleton, worm_i) for ell in el]
    )
    sample_params = np.array(
        [
            (t * (Ma - 1) / frame_max, s * (Mb - 1) / point_max)
            for t in worm_i
            for s in range(point_max)
        ]
    )

    spline_sheet.initializeFromPoints(sample_points, sample_params)
    control_points = [c.tolist() for c in spline_sheet.controlPoints]

    spline = sp.SplineSheet((Ma, Mb))
    spline.controlPoints = np.array(control_points)

    sampling_rate = frame_max
    num_isolines = point_max
    isolines = []

    for i in range(num_isolines):
        params = [
            (
                x * (spline.M[0] - 1) / (sampling_rate - 1),
                i * (spline.M[1] - 1) / (num_isolines - 1),
            )
            for x in range(sampling_rate)
        ]
        isolines.append(np.squeeze([spline.parametersToWorld(p) for p in params]))

    shape_list = np.array([np.flip(line, 1) for line in isolines])
    shape_list_array = np.array([shape_list[:, i, :] for i in range(sampling_rate)])

    return shape_list_array[:, :, [2, 1]]


def _return_tracked_data(Results_folder: str):
    results_path = Path(Results_folder).joinpath("skeletonNN.hdf5")

    with h5py.File(results_path, "r") as f:
        identities_list = [list(sublist) for sublist in f["identities_list"][:]]
        splines_list = [
            f["splines_list"][f"array_{i:05}"][:] for i in range(len(f["splines_list"]))
        ]
        splines_list = [np.array(arr) for arr in splines_list]

    return identities_list, splines_list


def _return_fill_gap(frame_window: np.array, skeleton: np.array):
    new_frame_window = np.arange(frame_window[0], frame_window[-1] + 1)
    new_skeletons = np.full(
        (new_frame_window.shape[0], skeleton.shape[1], skeleton.shape[2]), np.nan
    )

    # Fill in the skeletons
    for i, frame in enumerate(frame_window):
        index = frame - frame_window[0]
        new_skeletons[index] = skeleton[i]
    return new_skeletons, new_frame_window


def _process_skeletons(
    Results_folder, splines_list, identities_list, params_input, apply_spline=False
):
    """_summary_

    Args:
        Results_folder (_str_): Results folder where data will be saved
        splines_list (_list_): _List of skeletons_
        identities_list (_list_): _Identity of skeleton
        params_input (_class_): _Initialsed parameters _
        apply_spline (bool, optional): Apply spline. Defaults to False.
    """
    skeleton_folder = Path(Results_folder).joinpath("metadata_featuresN.hdf5")
    wormstats_header = tnn.wormstats()
    segment4angle = max(1, int(round(splines_list[0].shape[1] / 10)))
    unique_worm_IDS = set(x for sublist in identities_list for x in sublist)
    with tb.File(skeleton_folder, "w") as f:
        tab_time_series = f.create_table(
            "/",
            "timeseries_data",
            wormstats_header.header_timeseries,
            filters=TABLE_FILTERS,
        )
        tab_blob = f.create_table(
            "/", "blob_features", wormstats_header.blob_data, filters=TABLE_FILTERS
        )
        tab_traj = f.create_table(
            "/", "trajectories_data", wormstats_header.worm_data, filters=TABLE_FILTERS
        )
        worm_coords_array = {
            "skeleton": f.create_earray(
                "/",
                "skeleton",
                shape=(0, 49, 2),
                atom=tb.Float32Atom(),
                filters=TABLE_FILTERS,
            )
        }

        length_tab = 0
        if params_input.MW_mapping:
            fovsplitter = tnn._return_masked_image(params_input)

        for worm_index_joined in tqdm.tqdm(unique_worm_IDS, total=len(unique_worm_IDS)):
            worm_i, worm_j = np.where(
                pd.DataFrame(identities_list).values == worm_index_joined
            )

            try:
                if apply_spline:
                    skeleton = np.array(
                        [splines_list[x][y] for x, y in zip(worm_i, worm_j)]
                    )

                    skeleton = skeleton[~np.all(skeleton == 0, axis=(1, 2))]

                    if skeleton.size == 0:
                        continue

                    skeleton = correct_skeleton_orient(skeleton)
                    frame_window = np.array(
                        [
                            timestamp
                            for timestamp in (
                                worm_i * params_input.step_size
                                + params_input.start_frame
                                + params_input.min_frame
                            )
                        ]
                    ).reshape(-1)

                    segments_spline, frame_spline = (
                        segment_data_continuous_with_constraints(
                            skeleton,
                            frame_window,
                            params_input.step_size,
                            params_input.step_size,
                            params_input.spline_segments,
                        )
                    )

                    skeleton = np.concatenate(
                        [
                            _return_spline_skeleton(
                                skel, fr_stamp, params_input.time_param
                            )
                            for skel, fr_stamp in zip(segments_spline, frame_spline)
                            if len(fr_stamp) == params_input.spline_segments
                        ]
                    )
                    frame_window = np.concatenate(
                        [
                            range(fr_stamp.min(), fr_stamp.max())
                            for fr_stamp in frame_spline
                            if len(fr_stamp) == params_input.spline_segments
                        ]
                    )
                else:
                    skeleton = np.array(
                        [splines_list[x][y, :] for x, y in zip(worm_i, worm_j)]
                    )
                    skeleton = skeleton[~np.all(skeleton == 0, axis=(1, 2))]
                    if skeleton.size == 0:
                        continue
                    skeleton = correct_skeleton_orient(skeleton)
                    frame_window = (
                        worm_i * params_input.step_size
                        + params_input.start_frame
                        + params_input.min_frame
                    )

                skeleton, frame_window = _return_fill_gap(frame_window, skeleton)
                skeleton = correct_skeleton_orient(skeleton)

                is_switch_skel, _ = isWormHTSwitched(
                    skeleton,
                    segment4angle=segment4angle,
                    max_gap_allowed=params_input.max_gap_allowed,
                    window_std=params_input.window_std,
                    min_block_size=params_input.min_block_size,
                )
                skeleton[is_switch_skel] = skeleton[is_switch_skel, ::-1, :]
                worm_data = pd.DataFrame(
                    {
                        "frame_number": frame_window,
                        "skeleton_id": np.arange(
                            length_tab, length_tab + len(frame_window)
                        ),
                        "worm_index_joined": worm_index_joined,
                        "coord_x": skeleton[:, :, 0].mean(axis=1),
                        "coord_y": skeleton[:, :, 1].mean(axis=1),
                        "threshold": 84,
                        "has_skeleton": 1,
                        "roi_size": 102,
                        "area": 100,
                        "is_good_skel": 1,
                    }
                )
                timestamp = worm_data["frame_number"].values.astype(np.int32)
                feats = features.get_timeseries_features(
                    skeleton * params_input.microns_per_pixel,
                    timestamp=timestamp,
                    fps=params_input.expected_fps,
                ).astype(np.float32)
                feats.insert(0, "worm_index", worm_index_joined)
                feats.insert(
                    2,
                    "well_name",
                    (
                        fovsplitter.find_well_from_trajectories_data(worm_data)
                        if params_input.MW_mapping
                        else "n/a"
                    ),
                )
                feats["well_name"] = feats["well_name"].astype("S3")
                tab_traj.append(worm_data.to_records(index=False))
                set_unit_conversions(
                    tab_traj,
                    expected_fps=params_input.expected_fps,
                    microns_per_pixel=params_input.microns_per_pixel,
                )
                tab_time_series.append(feats.to_records(index=False))
                tab_blob.append(feats[["area", "length"]].to_records(index=False))
                worm_coords_array["skeleton"].append(skeleton)
                length_tab += len(worm_data)
                # if worm_index_joined == 10:
                #   break
                if params_input.MW_mapping:
                    fovsplitter.write_fov_wells_to_file(skeleton_folder)
            except Exception:
                continue


# %%

"""

Test and view the results 
"""
if __name__ == "__main__":
    input_vid = "/home/weheliye@cscdom.csc.mrc.ac.uk/behavgenom_mnt/Weheliye/Test/RawVideos/1.1_4_n2_6b_Set0_Pos0_Ch3_14012018_125942.hdf5"
    params_well = "/home/weheliye@cscdom.csc.mrc.ac.uk/behavgenom_mnt/Weheliye/Test/loopbio_rig_6WP_splitFOV_NN_20220202.json"

    apply_spline = True
    plot = False

    params_input, params_results = tnn._initialise_parameters(input_vid, params_well)

    bn = params_results["save_name"].stem
    # params_results["max_frame"] = 400
    if not params_results["save_name"].joinpath("skeletonNN.hdf5").exists():
        with tnn.selectVideoReader(input_vid) as store:
            tnn._detect_worm(
                store,
                params_input,
                **params_results,
            )

            tnn._track_worm(
                params_results["save_name"],
                memory=15,
                track_video_shape=(store.height, store.width),
            )

    identities_list, splines_list = _return_tracked_data(params_results["save_name"])

    _process_skeletons(
        params_results["save_name"],
        splines_list,
        identities_list,
        params_input,
        apply_spline=apply_spline,
    )

    if plot:
        Results_folder = params_results["save_name"]
        results_path = Path(Results_folder).joinpath("skeletonNN.hdf5")

        # Read predictions
        with h5py.File(results_path, "r") as f:
            predictions_list = [
                Predictions(
                    f["skeletonNN_w"][ds][:],
                    f["skeletonNN_s"][ds][:],
                    f["skeletonNN_p"][ds][:],
                )
                for ds in f["skeletonNN_p"]
            ]

        outdir_path = Results_folder.joinpath("plot/detect")
        outdir_path.mkdir(exist_ok=True, parents=True)
        with tnn.selectVideoReader(params_input.raw_fname) as store:
            t = np.arange(
                params_input.start_frame,
                1000,
                params_input.step_size,
            )
            for t, pred in zip(t, predictions_list):
                video = store.get_image(store.frame_min + t)[0]
                fig = plt.figure(figsize=(10.42, 10.42))
                plt.ylim(0, video.shape[0])
                plt.xlim(0, video.shape[1])
                plt.imshow(video, cmap="binary")

                for x in pred.w[:, 1]:
                    plt.plot(x[:, 0], x[:, 1], "-", color="red", linewidth=0.2)
                if t % 10 == 0:
                    figname = outdir_path.joinpath(f"{t:04d}.png")
                    fig.savefig(figname, pad_inches=0, bbox_inches="tight")
                plt.close(fig)


# %%


# def segment_data_continuous_with_constraints(
#     skeletons, frame_indices, step_size, max_gap=10, max_segment_length=10
# ):
#     """Segment skeleton data using a sliding window approach with constraints on gap and segment length."""
#     segments, frame_segments = [], []
#     current_segment, current_frames = [skeletons[0]], [frame_indices[0]]
#     min_segment_length = _min_segment_length(
#         step_size
#     )  # 3 if step_size >= 4 else max_segment_length

#     for start_idx in range(1, len(frame_indices)):
#         if (
#             frame_indices[start_idx] - frame_indices[start_idx - 1] > max_gap
#             or len(current_segment) >= max_segment_length
#         ):
#             if len(current_frames) >= 2:
#                 segments.append(np.array(current_segment))
#                 frame_segments.append(np.array(current_frames))

#             current_segment, current_frames = [], []

#             if (frame_indices[start_idx] - frame_indices[start_idx - 1]) <= max_gap:
#                 current_segment.append(skeletons[start_idx - 1])
#                 current_frames.append(frame_indices[start_idx - 1])

#         current_segment.append(skeletons[start_idx])
#         current_frames.append(frame_indices[start_idx])

#     # Add the last batch to segments
#     if len(current_frames) >= min_segment_length:
#         segments.append(np.array(current_segment))
#         frame_segments.append(np.array(current_frames))

#     # Interpolate segments to ensure length is 10
#     interpolated_segments, interpolated_frame_segments = [], []
#     for segment, frames in zip(segments, frame_segments):
#         if min_segment_length < len(segment) < max_segment_length:
#             interpolated_segment, interpolated_frames = linear_interpolate(
#                 segment, frames
#             )
#             interpolated_segments.append(interpolated_segment)
#             interpolated_frame_segments.append(interpolated_frames)
#         else:
#             interpolated_segments.append(segment)
#             interpolated_frame_segments.append(frames)

#     return interpolated_segments, interpolated_frame_segments


# # %
# Results_folder = params_results["save_name"]
# skeleton_folder = Path(Results_folder).joinpath("metadata_featuresN.hdf5")
# wormstats_header = tnn.wormstats()
# segment4angle = max(1, int(round(splines_list[0].shape[1] / 10)))
# unique_worm_IDS = set(x for sublist in identities_list for x in sublist)


# for worm_index_joined in tqdm.tqdm(unique_worm_IDS, total=len(unique_worm_IDS)):
#     worm_i, worm_j = np.where(pd.DataFrame(identities_list).values == worm_index_joined)

#     if worm_index_joined == 3:
#         break
#     if apply_spline:
#         skeleton = np.array([splines_list[x][y] for x, y in zip(worm_i, worm_j)])

#         skeleton = skeleton[~np.all(skeleton == 0, axis=(1, 2))]

#         skeleton = correct_skeleton_orient(skeleton)
#         frame_window = np.array(
#             [
#                 timestamp
#                 for timestamp in (
#                     worm_i * params_input.step_size
#                     + params_input.start_frame
#                     + params_input.min_frame
#                 )
#             ]
#         ).reshape(-1)
#         # if (
#         #     params_input.expected_fps // params_input.skip_frame >= 5
#         # ):  # params_input.skip_frame >= 5:
#         segments_spline, frame_spline = segment_data_continuous_with_constraints(
#             skeleton,
#             frame_window,
#             params_input.step_size,
#             params_input.step_size,
#             params_input.spline_segments,
#         )
#         # else:
#         #     segments_spline, frame_spline = (
#         #         segment_data_continuous_with_constraints_low_fps(
#         #             skeleton,
#         #             frame_window,
#         #             params_input.skip_frame,
#         #             params_input.spline_segments,
#         #         )
#         #     )

#         skeleton = np.concatenate(
#             [
#                 _return_spline_skeleton(skel, fr_stamp, 6)
#                 for skel, fr_stamp in zip(segments_spline, frame_spline)
#                 if len(fr_stamp) == params_input.spline_segments
#             ]
#         )
#         frame_window = np.concatenate(
#             [
#                 range(fr_stamp.min(), fr_stamp.max())
#                 for fr_stamp in frame_spline
#                 if len(fr_stamp) == params_input.spline_segments
#             ]
#         )
# # %%
