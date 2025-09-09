# %%


from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange

import deeptangle as dt
import tierpsynn as tnn
from deeptangle.predict import Predictions, non_max_suppression

# %%%


def non_max_suppression(p, threshold=0.5, overlap_threshold=0.5, cutoff=96):
    p = jax.tree_util.tree_map(lambda x: np.array(x), p)
    non_suppressed_p = dt.non_max_suppression(
        p, threshold=threshold, overlap_threshold=overlap_threshold, cutoff=cutoff
    )
    return jax.tree_util.tree_map(lambda x: x[non_suppressed_p], p)


def predict_in_batches(x, forward_fn, state, params_input):
    """ "
    Predict the images in batches

    """
    trim_frames = int(len(x) - len(x) % params_input.num_batches)
    new_shape = (params_input.num_batches, -1, *x[0].shape)
    batched_X = jnp.reshape(x[:trim_frames], new_shape)
    scan_predict_fn = lambda _, u: (None, dt.predict(forward_fn, state, u))
    _, y = jax.lax.scan(scan_predict_fn, init=None, xs=batched_X)
    y = jax.tree_util.tree_map(lambda u: jnp.reshape(u, (-1, *u.shape[2:])), y)
    y = jax.tree_util.tree_map(np.array, y)
    predictions = list(map(dt.Predictions, *y))
    predictions = [
        non_max_suppression(
            p,
            params_input.threshold,
            params_input.overlap_threshold,
            params_input.cutoff,
        )
        for p in predictions
    ]

    return predictions


def _detect_worm(
    store,
    params_input,
    save_name,
    forward_fn,
    state_single,
    min_frame,
    max_frame,
    max_frame_init,
    bgd,
    pad_bottom,
    pad_right,
    scale_factor,
):
    with h5py.File(Path(save_name).joinpath("skeletonNN.hdf5"), "w") as f:
        w_group = f.create_group("skeletonNN_w")
        p_group = f.create_group("skeletonNN_p")
        s_group = f.create_group("skeletonNN_s")

        batch, frame_list = [], []

        clip = np.array([store.read()[1] for _ in range(min_frame, max_frame_init)])
        bn = Path(save_name).stem  # parent.name

        if scale_factor:
            clip_rescaled = jax.image.resize(
                clip[0],
                [
                    int(clip[0].shape[0] * scale_factor),
                    int(clip[0].shape[1] * scale_factor),
                ],
                method="cubic",
            )

            scale_x, scale_y = (
                clip.shape[1] / clip_rescaled.shape[0],
                clip.shape[2] / clip_rescaled.shape[1],
            )

        for time_stamp, frame_number in enumerate(
            trange(
                0,
                max_frame - max_frame_init - params_input.step_size,
                params_input.step_size,
                desc=bn,
            )
        ):
            try:
                clip_inverted = (
                    255 - clip[:: params_input.skip_frame, :]
                    if params_input.is_light
                    else clip[:: params_input.skip_frame, :]
                )
                th = (
                    tnn._adaptive_thresholding(clip_inverted, params_input)
                    * clip_inverted
                )

                if params_input.bgd_removal:
                    th *= (clip_inverted - bgd) > params_input.SVD_thres

                if scale_factor:
                    th = jax.image.resize(
                        th,
                        [
                            th.shape[0],
                            int(th.shape[1] * scale_factor),
                            int(th.shape[2] * scale_factor),
                        ],
                        method="cubic",
                    )

                th = jnp.pad(
                    th,
                    ((0, 0), (0, pad_bottom), (0, pad_right)),
                    mode="constant",
                    constant_values=0,
                )

                if time_stamp % params_input.num_batches == 0 and time_stamp != 0:
                    batch = jnp.asarray(batch)
                    predictions = predict_in_batches(
                        batch, forward_fn, state_single, params_input
                    )
                    for Frame, preds in zip(frame_list, predictions):
                        dataset_name = f"frame_number_{Frame:05}"
                        w, s, p = preds
                        if w.size == 0:
                            w = np.zeros([1, 3, 49, 2])
                            p = np.zeros([1, 8])
                        if scale_factor:
                            w[:, :, :, 0] *= scale_x
                            w[:, :, :, 1] *= scale_y

                        w_group.create_dataset(dataset_name, data=w, compression="gzip")
                        p_group.create_dataset(dataset_name, data=p, compression="gzip")
                        s_group.create_dataset(dataset_name, data=s, compression="gzip")
                    batch, frame_list = [], []

                frame_list.append(int(frame_number + params_input.start_frame))
                batch.append((jnp.asarray(th / 255, dtype=np.float32)))
                # batch.append( jnp.asarray(th, dtype=jnp.float32) )  # Normalizing and adding to batch
                small_clip = np.array(
                    [store.read()[1] for _ in range(params_input.step_size)]
                )
                clip = np.concatenate(
                    (clip[params_input.step_size :, :], small_clip), axis=0
                )
            except Exception:
                continue


def _track_worm(
    Results_folder: str,
    window=50,
    memory=15,
    adaptive_step=0.1,
    track_video_shape=None,
):
    """
    Post processing stage
    """
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

    # Process and save results
    with h5py.File(results_path, "a") as f:
        identities_list, splines_list = dt.identity_assignment(
            predictions_list, memory, adaptive_step=adaptive_step, apply_spline=False
        )

        identities_list, splines_list = dt.merge_tracks(
            identities_list, splines_list, window=window, framesize=track_video_shape[1]
        )

        # Save identities list
        identity_set = f.create_dataset(
            "identities_list",
            (len(identities_list),),
            compression="gzip",
            dtype=h5py.special_dtype(vlen=np.dtype("int")),
        )
        for i, sublist_id in enumerate(identities_list):
            identity_set[i] = sublist_id

        # Save splines list
        spline_group = f.create_group("splines_list")
        for i, array in enumerate(splines_list):
            spline_group.create_dataset(f"array_{i:05}", data=array, compression="gzip")

    # return identities_list, splines_list


# %%

if __name__ == "__main__":
    input_vid = "/home/weheliye@cscdom.csc.mrc.ac.uk/behavgenom_mnt/Weheliye/Test/RawVideos/1.1_4_n2_6b_Set0_Pos0_Ch3_14012018_125942.hdf5"
    params_well = "/home/weheliye@cscdom.csc.mrc.ac.uk/behavgenom_mnt/Weheliye/Test/loopbio_rig_6WP_splitFOV_NN_20220202.json"

    params_input, params_results = tnn._initialise_parameters(input_vid, params_well)

    bn = params_results["save_name"].stem
    params_results["max_frame"] = 400

    with tnn.selectVideoReader(input_vid) as store:
        _detect_worm(store, params_input, **params_results)

        if not tnn.is_hdf5_empty(
            Path(params_results["save_name"]).joinpath("skeletonNN.hdf5")
        ):
            _track_worm(
                params_results["save_name"],
                memory=15,
                track_video_shape=(store.height, store.width),
            )


# %%
