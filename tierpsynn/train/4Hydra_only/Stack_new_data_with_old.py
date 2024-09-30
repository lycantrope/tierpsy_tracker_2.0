# %%
# Importing necessary libraries
%matplotlib qt

from pathlib import Path
import h5py
import numpy as np
import tqdm
import cv2
import matplotlib.pyplot as plt
import itertools
import jax.numpy as jnp
from scipy.interpolate import interp1d

# Set the default colormap for matplotlib to 'gray'
plt.rcParams['image.cmap'] = 'gray'


# %%
# Function to normalize label data by resampling points
def normalize_label_data(label_data, num_points=49):
    """
    Resample the points in the label data to ensure uniform spacing.

    Args:
        label_data (ndarray): Input label data of shape (N, M, T, 2).
        num_points (int): Number of points to resample to.

    Returns:
        ndarray: Label data with resampled points.
    """
    for i, j in itertools.product(range(label_data.shape[0]), range(label_data.shape[1])):
        x = label_data[i, j, :, 0]
        y = label_data[i, j, :, 1]

        # Calculate the cumulative distance along the curve
        distance = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0) ** 2 + np.ediff1d(y, to_begin=0) ** 2))
        distance /= distance[-1]  # Normalize the distance to [0, 1]

        # Interpolation functions for x and y
        fx = interp1d(distance, x)
        fy = interp1d(distance, y)

        # Generate regular spaced points between [0, 1]
        alpha = np.linspace(0, 1, num_points)
        x_regular, y_regular = fx(alpha), fy(alpha)

        # Update the label data with resampled points
        label_data[i, j, :, 0] = x_regular
        label_data[i, j, :, 1] = y_regular

    return label_data


# %%
# Function to apply image augmentation (rotation)
def augment_data(img, label, k=1):
    """
    Rotate the image and label by 90*k degrees.

    Args:
        img (ndarray): Image array of shape (C, H, W).
        label (ndarray): Label array of shape (N, M, T, 2).
        k (int): Number of 90-degree rotations (k=1 rotates by 90 degrees).

    Returns:
        tuple: Rotated image and label arrays.
    """
    # Rotation matrix for 90*k degree rotation around image center (512x512 assumed)
    M = cv2.getRotationMatrix2D((512 / 2, 512 / 2), 90 * k, 1)

    # Rotate the image using jax.numpy for better performance
    img = jnp.rot90(img, k, axes=(1, 2))

    # Rotate each label coordinate using the rotation matrix
    for i, j in itertools.product(range(label.shape[0]), range(label.shape[1])):
        label[i, j, :, :] = (label[i, j, :, :] @ M[:2, :2].T) + M[:, -1][:, None].T

    return img, label


# %%
# Function for adaptive thresholding
def adaptive_thresholding(img, blocksize=31, constant=15):
    """
    Apply adaptive thresholding to each channel in the image.

    Args:
        img (ndarray): Image array of shape (C, H, W).
        blocksize (int): Size of the neighborhood used for thresholding.
        constant (int): Constant subtracted from the mean during thresholding.

    Returns:
        ndarray: Binary thresholded image array.
    """
    # Convert image to uint8 and invert
    img = (255 - img * 255).astype(np.uint8)

    # Apply adaptive thresholding to each channel
    thresholded = np.array([
        cv2.adaptiveThreshold(img[i, :], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blocksize, constant) == 0
        for i in range(img.shape[0])
    ])

    return thresholded


# %%
# File paths for training data
filepath_training = Path('Hand_annotated_worm_pose.hdf5')
filepath_training_stacked = Path('Final_mixed_train_data_wt_hand_ann.hdf5')

# %%
# Process and augment data from the training file
with h5py.File(filepath_training, 'r+') as f, h5py.File(filepath_training_stacked, 'a') as f_s:
    arrays_group = f['x_train']
    y_arrays_group = f['y_train']
    xtrain_stack = f_s['x_train']
    ytrain_stack = f_s['y_train']

    # Start counting from the next available file index
    k_files = int(list(xtrain_stack)[-1].split('_')[-1]) + 1
    print(f"Starting file index: {k_files}")

    # Iterate over training data and process
    for i, (dataset_name_x, dataset_name_y) in tqdm.tqdm(enumerate(zip(arrays_group, y_arrays_group))):
        # Load and preprocess image
        X = arrays_group[dataset_name_x][0, :]  # Shape (C, H, W)
        X = (255 - X) / 255  # Normalize the image
        th = adaptive_thresholding(X)  # Apply thresholding
        X *= th  # Mask image with threshold

        # Load and preprocess labels
        Y = y_arrays_group[dataset_name_y][0, :]  # Shape (N, M, T, 2)
        Y = normalize_label_data(Y)  # Normalize label data

        # Save processed data to the new file
        dataset_name = f'array_{k_files:05}'
        xtrain_stack.create_dataset(dataset_name, data=X, compression='gzip')
        ytrain_stack.create_dataset(dataset_name, data=Y, compression='gzip')

        # Increment the file index
        k_files += 1
