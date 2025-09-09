# %% Imports
%matplotlib qt
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import deeptangle as dt
import imgstore
import cv2
import h5py
import tqdm
import random
from deeptangle import utils, build_model, checkpoints
from deeptangle.predict import non_max_suppression as nms
from skimage.transform import resize
from tierpsy.analysis.split_fov.FOVMultiWellsSplitter import FOVMultiWellsSplitter
from tierpsy.helper.params.tracker_param import SplitFOVParams
from tierpsy.analysis.compress.selectVideoReader import selectVideoReader
import os
jax.config.update('jax_platform_name', 'gpu')
print(jax.local_devices())

# %% Parameters
model_path = Path('/home/weheliye@cscdom.csc.mrc.ac.uk/Desktop/Behavgenom_repo/DT_C/tierpsynn/extras/models/parameters_Final_mixed_train_data_wt_hand_ann_pca_92_epochs_250_2024-04-19_96_cutoff')
nframes = 11
n_suggestions = 8
latent_dim = 8
expected_fps = 25
microns_per_pixel = 12.4
max_gap_allowed = max(1, int(expected_fps // 2))
window_std = max(int(round(expected_fps)), 5)
min_block_size = max(int(round(expected_fps)), 5)
num_batches =10

# %% Functions
def _return_masked_image(raw_fname, px2um=microns_per_pixel):
    
    json_fname = Path('Path2JSON/HYDRA_96WP_UPRIGHT.json')

    splitfov_params = SplitFOVParams(json_file=json_fname)
    shape, edge_frac, sz_mm = splitfov_params.get_common_params()
    uid, rig, ch, mwp_map = splitfov_params.get_params_from_filename(
        raw_fname)
    px2um = px2um

    # read image
    with selectVideoReader(str(raw_fname)) as vid:
        status, img = vid.read_frame(0)

        fovsplitter = FOVMultiWellsSplitter(
            img,
            microns_per_pixel=px2um,
            well_shape=shape,
            well_size_mm=sz_mm,
            well_masked_edge=edge_frac,
            camera_serial=uid,
            rig=rig,
            channel=ch,
            wells_map=mwp_map)
        #fig = fovsplitter.plot_wells()
    return  fovsplitter.get_wells_data()

def load_video(raw_videos):
    return imgstore.new_for_filename(raw_videos)

def load_model_weights(origin_dir, broadcast=False):
    path = Path(origin_dir)
    A = jnp.load(path.joinpath('eigenworms_transform.npy').open('rb'))
    forward_fn = build_model(A, n_suggestions, latent_dim, nframes)
    state = checkpoints.restore(origin_dir, broadcast=broadcast)
    return forward_fn, state

def adaptive_threshold(img, blocksize=15, constant=1):
    return np.array([cv2.adaptiveThreshold(255 - img[j, :], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blocksize, constant) == 0 for j in range(img.shape[0])])

def SVD_bgd(store, max_frame, skip_frame, scale=1):
    clip = 255 - np.array([store.get_image(store.frame_min + frame)[0] for frame in range(0, max_frame, skip_frame)])
    clip_resized = resize(clip, [clip.shape[0], int(clip.shape[1] / scale), int(clip.shape[2] / scale)])
    U, s, V = np.linalg.svd(clip_resized.reshape(clip_resized.shape[0], -1), full_matrices=False)
    low_rank = np.expand_dims(U[:, 0], 1) * s[0] * np.expand_dims(V[0, :], 0)
    return resize(low_rank.reshape(clip_resized.shape), clip.shape)


def non_max_suppression(p, threshold=0.5, overlap_threshold=0.5, cutoff=96):
    p = jax.tree_util.tree_map(lambda x: np.array(x), p)
    non_suppressed_p = dt.non_max_suppression(p, threshold=threshold, overlap_threshold=overlap_threshold, cutoff=cutoff)
    return jax.tree_util.tree_map(lambda x: x[non_suppressed_p], p)

def predict_in_batches(x, forward_fn, state):

            trim_frames = int(len(x) - len(x) % num_batches)
            new_shape = (num_batches, -1, *x[0].shape)
            batched_X = jnp.reshape(x[:trim_frames], new_shape)
            scan_predict_fn = lambda _, u: (None, dt.predict(forward_fn, state, u))
            _, y = jax.lax.scan(scan_predict_fn, init=None, xs=batched_X)
            y = jax.tree_util.tree_map(lambda u: jnp.reshape(u, (-1, *u.shape[2:])), y)
            y = jax.tree_util.tree_map(np.array, y)
            predictions = list(map(dt.Predictions, *y))
            predictions = [non_max_suppression(p) for p in predictions]
                
     
            return predictions
        


def process_video(bgd, save_path, i_x, i_y, store, min_frame, max_frame, step_size, inter_skip, start_frame, well_number):
    save_file = save_path.joinpath('skeletonNN.hdf5')
    clip = np.array([(store.get_next_image()[0])[i_x:i_x+512, i_y:i_y+512] for _ in range(min_frame, max_frame)])

    with h5py.File(save_file, 'a' if save_file.exists() else 'w') as f:
        y_group = f.create_group('y_train') if 'y_train' not in f else f['y_train']
        x_group = f.create_group('x_train') if 'x_train' not in f else f['x_train']
        bgd_th_group = f.create_group('x_real_train') if 'x_real_train' not in f else f['x_real_train']
        batch, batch_original, frame_list = [], [], []
        
        for time_stamp, frame_number in enumerate(tqdm.trange(0, 1000 - max_frame - inter_skip, inter_skip)):
            clip_inverted = 255 - clip[::step_size, :]
            th = adaptive_threshold(clip_inverted) * clip_inverted
         


            if time_stamp % 10 == 0 and batch:
                predictions = predict_in_batches(jnp.asarray(batch), forward_fn, state_single)
                for Frame, preds, clip_org, bgd_th in zip(frame_list, predictions, batch_original, batch):
                    dataset_name = f'WN_{well_number:05}_FN_{Frame:05}'
                    print(dataset_name)
                    w, _, _ = preds
                    if w.size:
                        y_group.create_dataset(dataset_name, data=w[None, :], compression='gzip')
                        x_group.create_dataset(dataset_name, data=clip_org[None, :], compression='gzip')
                        bgd_th_group.create_dataset(dataset_name, data=bgd_th[None, :], compression='gzip')

                batch, batch_original, frame_list = [], [], []

            batch.append(jnp.asarray(th / 255, dtype=np.float32))
            frame_list.append(frame_number + start_frame)
            batch_original.append(clip[::step_size, :])
            small_clip =  np.array([(store.get_next_image()[0])[i_x:i_x+(512),i_y:i_y+(512)] for _ in range(inter_skip)])
            clip = np.concatenate((clip[inter_skip:,:], small_clip), axis=0 )

# %% Main Execution
forward_fn, state = load_model_weights(model_path)
state_single = utils.single_from_sharded(state)

main_folder = 'Thick_lawn'
people_list = ['John']
step_sizes = [5, 5]
k =0
for person in people_list:
    input_vid = Path(main_folder).joinpath(person)
    save_path = input_vid.joinpath('Training_set')
    save_path.mkdir(exist_ok=True, parents=True)
    
    yaml_files = [os.path.join(root, name) for root, dirs, files in os.walk(input_vid) for name in files if name.endswith('.yaml')]
   
    for src_file in (yaml_files):
        data = _return_masked_image(src_file)
        data['x_edge'] = ((data['x_min'] + data['x_max']) / 2) - 256
        data['y_edge'] = ((data['y_min'] + data['y_max']) / 2) - 256
        
        store = load_video(src_file)
        bgd = 0#SVD_bgd(store, store.frame_count, 500).min(axis=0) * 255
        
        for _, row in data.iterrows():
            k+=1
            i, j = int(row['y_edge']), int(row['x_edge'])
            step_size_choice = random.choice(step_sizes)
            store = imgstore.new_for_filename(str(src_file))
            process_video(bgd, save_path, i, j, store, 0, nframes * step_size_choice, step_size_choice, 50, 0, k)

# %%
