# %%
import argparse
import warnings

import tierpsynn as tnn

warnings.filterwarnings("ignore")


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


def main(input_vid, params_well, apply_spline, track_detect, get_features):
    if track_detect:
        """
         Initialise the parameters and detect and track worms 

        """

        params_input, params_results = tnn._initialise_parameters(
            input_vid, params_well
        )

        with tnn.selectVideoReader(input_vid) as store:
            tnn._detect_worm(
                store,
                params_input,
                **params_results,
            )
            tnn._track_worm(
                params_results["save_name"],
                memory=15,
                window=params_input.tracking_window_time,
                track_video_shape=(store.height, store.width),
            )

    if get_features:
        """_
        Get Feature summaries that are compatible with tierpsy tracker viewer 
        """
        params_input, save_name = tnn._initialise_parameters_features(
            input_vid, params_well
        )

        identities_list, splines_list = tnn._return_tracked_data(save_name)
        tnn._process_skeletons(
            save_name,
            splines_list,
            identities_list,
            params_input,
            apply_spline=apply_spline,
        )


def process_main_local():
    parser = argparse.ArgumentParser(description="Track and segment worms in a video.")
    parser.add_argument("--input_vid", type=str, help="Path to the input video file")
    parser.add_argument("--params_well", type=str, help="Path to the params well file")
    parser.add_argument(
        "--apply_spline", type=str2bool, default=False, help="Apply spline"
    )
    parser.add_argument(
        "--track_detect", type=str2bool, default=False, help="Track and detect"
    )
    parser.add_argument(
        "--get_features", type=str2bool, default=False, help="post process"
    )

    args = parser.parse_args()
    main(
        args.input_vid,
        args.params_well,
        args.apply_spline,
        args.track_detect,
        args.get_features,
    )


if __name__ == "__main__":
    process_main_local()
