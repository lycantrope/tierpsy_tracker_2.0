import os

from tierpsynn.analysis.detect_track_process import _detect_worm, _track_worm
from tierpsynn.analysis.features_spline import _process_skeletons, _return_tracked_data
from tierpsynn.extras.Wormstats import wormstats
from tierpsynn.helper.pre_processing import (
    _adaptive_thresholding,
    _initialise_parameters,
    _initialise_parameters_features,
    _return_masked_image,
    is_hdf5_empty,
    selectVideoReader,
)

__all__ = [
    "selectVideoReader",
    "_adaptive_thresholding",
    "_initialise_parameters",
    "is_hdf5_empty",
    "_return_masked_image",
    "_detect_worm",
    "_track_worm",
    "wormstats",
    "_process_skeletons",
    "_initialise_parameters_features",
    "_return_tracked_data",
]

base_path = os.path.dirname(__file__)

MODEL_FILES_DIR = os.path.abspath(
    os.path.join(
        base_path,
        "extras",
        "models",
        "weights",
    )
)


loopbio_6WP = os.path.abspath(
    os.path.join(base_path, "extras/configs/loopbio_rig_6WP_splitFOV_NN_20220202.json")
)

loopbio_24WP = os.path.abspath(
    os.path.join(base_path, "extras/configs/loopbio_rig_24WP_splitFOV_NN_20220202.json")
)

loopbio_96WP = os.path.abspath(
    os.path.join(base_path, "extras/configs/loopbio_rig_96WP_splitFOV_NN_20220202.json")
)


MWP_PATH = os.path.abspath(os.path.join(base_path, "extras", "configs"))
