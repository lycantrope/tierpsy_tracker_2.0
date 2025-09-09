# %%

import json
import warnings

import tierpsynn as tnn

warnings.simplefilter("always", UserWarning)

# %%


class Config:
    def __init__(self, well_type, raw_fname):
        self.raw_fname = raw_fname
        with tnn.selectVideoReader(self.raw_fname) as input_video:
            self.well_type = well_type
            self.params_in_file = self._return_parameters()
            self.expected_fps = self.params_in_file["expected_fps"]
            if not self.expected_fps:
                try:
                    self.expected_fps = int(input_video.fps)  # self._expected_fps()
                except Exception:
                    self.expected_fps = 1
                    warnings.warn(
                        "Cannot determine the FPS; therefore, assigned a value of 1. "
                        "If you know your FPS value, please modify the JSON parameters file for the expected FPS.",
                        UserWarning,
                    )
            self.input_video = input_video

        self.step_size = self.params_in_file["step_size"]
        self.threshold = self.params_in_file["threshold"]
        self.overlap_threshold = self.params_in_file["overlap_threshold"]
        self.cutoff = self.params_in_file["cutoff"]
        self.skip_frame = self.params_in_file["skip_frame"]
        self.spline_segments = self.params_in_file["spline_segments"]
        if not self.skip_frame:
            self.skip_frame = max(1, int(self.expected_fps / 5))

        if not self.step_size:
            self.step_size = max(1, int(self.expected_fps / 5))

        self.nframes = self.params_in_file["nframes"]
        self.n_suggestions = self.params_in_file["n_suggestions"]
        self.latent_dim = self.params_in_file["latent_dim"]
        self.microns_per_pixel = self.params_in_file["microns_per_pixel"]
        self.MW_mapping = self.params_in_file["MWP_mapping"]
        self.model = self.params_in_file["model_path"]
        if self.model == "default":
            self.model = tnn.MODEL_FILES_DIR
        self.is_light = self.params_in_file["is_light_background"]
        self.block_size = self.params_in_file["thresh_C"]
        self.Constant = self.params_in_file["thresh_block_size"]
        self.SVD_thres = self.params_in_file["SVD_threshold"]
        self.min_frame = self.params_in_file["min_frame"]
        self.max_frame = self.params_in_file["max_frame"]
        self.num_batches = self.params_in_file["num_batches"]
        self.max_gap_allowed = max(1, int(self.expected_fps // 2))
        self.window_std = max(int(round(self.expected_fps)), 5)
        self.min_block_size = max(int(round(self.expected_fps)), 5)
        self.max_frame_init = self.nframes * self.skip_frame
        self.start_frame = 5 * self.skip_frame
        self.bgd_removal = self.params_in_file["remove_background"]
        self.time_param = self.params_in_file["Time_param"]
        self.tracking_window_time = self.params_in_file["Tracking_window_time"]
        self.maximum_spline_gap = self.params_in_file["Maximum_spline_segment_gaps"]
        if not self.maximum_spline_gap:
            self.maximum_spline_gap = self.step_size * 2
        self.scale_factor = self.params_in_file["scale_factor"]

        # print(self.maximum_spline_gap)

    def read_params(self, json_file=""):
        if json_file:
            with open(json_file) as fid:
                params_in_file = json.load(fid)
        return params_in_file

    def _return_parameters(self):
        config_files = {
            24: tnn.loopbio_24WP,
            96: tnn.loopbio_96WP,
            6: tnn.loopbio_6WP,
        }

        if self.well_type in config_files:
            params_in_file = self.read_params(config_files[self.well_type])
        else:
            params_in_file = self.read_params(self.well_type)

        return params_in_file


# %%

if __name__ == "__main__":
    # input_vid = '/home/weheliye@cscdom.csc.mrc.ac.uk/behavgenom_mnt/Weheliye/Paper_4_Andre/Data/camb_data/Phenix/Aggregation/RawVideos/1.1_4_n2_6b_Set0_Pos0_Ch3_14012018_125942.hdf5'

    input_vid = "/home/weheliye@cscdom.csc.mrc.ac.uk/behavgenom_mnt/Weheliye/Paper_4_Andre/Data/transfer_2833382_files_c35b8490/RawVideos/MultiwormTest.mp4"
    # input_vid = "/home/weheliye@cscdom.csc.mrc.ac.uk/behavgenom_mnt/Weheliye/Paper_4_Andre/Data/hydraAggregation/RawVideos/20210301/sixwellaggregation_bluelight_20210301_175959.22956806/metadata.yaml"
    params_well = 96
    params_input = Config(params_well, input_vid)
    print(params_input.expected_fps)

# %%
