# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:44:07 2015

@author: ajaver
"""

import json
import os
import warnings
import numpy as np

#get default parameters files
from tierpsy import DFLT_PARAMS_PATH, DFLT_PARAMS_FILES
from tierpsy import DFLT_SPLITFOV_PARAMS_PATH, DFLT_SPLITFOV_PARAMS_FILES
from .docs_tracker_param import default_param, info_param, valid_options
from .docs_analysis_points import dflt_analysis_points, dlft_analysis_type, deprecated_analysis_alias

#deprecated variables that will be ignored
deprecated_fields = [
                    'has_timestamp',
                    'min_displacement',
                    'fps_filter',
                    'traj_bgnd_buff_size',
                    'traj_bgnd_frame_gap'
                    ]

#the dict key are the old names and the value the new name
deprecated_alias = {
    'fps': 'expected_fps',
    'threshold_factor': 'worm_bw_thresh_factor',
    'is_invert_thresh' : 'is_light_background',
    'is_fluorescence' : 'is_light_background',
    'min_length' : 'traj_min_box_width',
    'min_box_width': 'traj_min_box_width',
    'max_area' : 'mask_max_area',
    'save_int_maps': 'int_save_maps',
    'is_extract_metadata':'is_extract_timestamp',
    }


deprecated_splitFOV_params = [
    "MWP_total_n_wells",
    "MWP_whichsideup",
    "MWP_well_shape",
    # "MWP_well_masked_edge",
]

def get_dflt_sequence(analysis_type):
    assert analysis_type in valid_options['analysis_type']
    analysis_checkpoints = dflt_analysis_points[analysis_type].copy()
    return analysis_checkpoints


def fix_deprecated(param_in_file):
    '''
    basically look for deprecated or changed names and corrected with the newest versions them.
    '''
    corrected_params = {}
    for key in param_in_file:
        if key in deprecated_fields:
            #ignore deprecated fields
            continue
        elif key in deprecated_alias:
            #rename deprecated alias
            corrected_params[deprecated_alias[key]] = param_in_file[key]

        elif key == 'min_area':
            #special case of deprecated alias
            corrected_params['mask_min_area'] = param_in_file['min_area']
            corrected_params['traj_min_area'] = param_in_file['min_area']/2
        elif key == 'analysis_type':
            #correct the name of a deprecated analysis types
            vv = param_in_file['analysis_type']
            corrected_params['analysis_type'] = deprecated_analysis_alias[vv] if vv in deprecated_analysis_alias else vv

        elif key == 'filter_model_name':
            # corrected_params['use_nn_filter'] = len(param_in_file['filter_model_name']) > 0 #set to true if it is not empty
            # maintain legacy deprectaion handling (above line)
            corrected_params['nn_filter_to_use'] = 'tensorflow_default'
        elif key == 'use_nn_filter':
            # legacy behaviour before pytorch NN
            if param_in_file[key] is True:
                corrected_params['nn_filter_to_use'] = 'tensorflow_default'
            elif param_in_file[key] is False:
                corrected_params['nn_filter_to_use'] = 'none'
            else:
                # what if somebody has not used the right name in the json?
                warnings.warn('bool parameter "use_nn_filter" is deprecated.'
                              + ' Use str parameter "nn_filter_to_use".')
                corrected_params['nn_filter_to_use'] = param_in_file[key]
        else:
            corrected_params[key] = param_in_file[key]
    # import pdb; pdb.set_trace()
    return corrected_params


def fix_types(param_in_file):
    '''
    Using the GUI to set parameters leads to MWP_total_n_wells to be a str
    rather than an int. This function fixes this problem and can be used for
    any other that we may encounter
    '''
    if 'MWP_total_n_wells' in param_in_file.keys():
        if isinstance(param_in_file['MWP_total_n_wells'], str):
            param_in_file['MWP_total_n_wells'] = int(
                    param_in_file['MWP_total_n_wells'])

    return param_in_file


def fix_deprecated_splitFOV_params(param_in_file):
    """
    if no json has been specified to split the field of view, and instead
    the deprecated way of specifying parameters is used, convert to the new
    usage (only for 96wp, upright, square wells)
    """
    if ('MWP_mapping' not in param_in_file) and any(
            k in param_in_file for k in deprecated_splitFOV_params):
        print(
            'Deprecated MWP parameters!\n'
            'Write a json file for splitFOV params, or set\n'
            '"MWP_mapping": ""\n'
            'in the parameters\' file')
        if all(k in param_in_file for k in deprecated_splitFOV_params):
            if (param_in_file["MWP_total_n_wells"] == 96 and
                    param_in_file["MWP_whichsideup"] == "upright" and
                    param_in_file["MWP_well_shape"] == "square"):
                param_in_file['MWP_mapping'] = 'HYDRA_96WP_UPRIGHT.json'
            if (param_in_file["MWP_total_n_wells"] == 24 and
                    param_in_file["MWP_whichsideup"] == "upright" and
                    param_in_file["MWP_well_shape"] == "circle"):
                param_in_file['MWP_mapping'] = 'HYDRA_24WP_UPRIGHT.json'
    # remove deprecated
    for key in deprecated_splitFOV_params:
        if key in param_in_file:
            param_in_file.pop(key)
    return param_in_file


def read_params(json_file=''):
    '''
    Read input, and assign defults for the missing values.
    '''
    input_param = default_param.copy()
    if json_file:
        with open(json_file) as fid:
            param_in_file = json.load(fid)
        param_in_file = fix_deprecated(param_in_file)
        param_in_file = fix_types(param_in_file)
        param_in_file = fix_deprecated_splitFOV_params(param_in_file)

        for key in param_in_file:
            if key in input_param:
                input_param[key] = param_in_file[key]
            else:
                raise ValueError('Parameter "{}" is not a valid parameter. Change its value in file "{}"'.format(key, json_file))

            if key in valid_options:

                is_in_list = input_param[key] in valid_options[key]
                try:
                    is_in_list = is_in_list or (
                        int(input_param[key]) in valid_options[key])
                except:
                    pass
                if not is_in_list:
                    raise ValueError('Parameter "{}" is not in the list of valid options {}'.format(param_in_file[key],valid_options[key]))

        if not input_param['analysis_checkpoints']:
            input_param['analysis_checkpoints'] = get_dflt_sequence(input_param['analysis_type'])

    return input_param



#AFTER THE LAST MODIFICATION I DON'T THINK THIS SHOULD BE A OBJECT,
#BUT I WOULD LEAVE IT LIKE THIS FOR THE MOMENT FOR BACK COMPATIBILITY
class TrackerParams:

    def __init__(self, json_file=''):
        #If the json_file is in the extras/param_file directory add the full path
        if json_file in DFLT_PARAMS_FILES:
            json_file = os.path.join(DFLT_PARAMS_PATH, json_file)

        self.json_file = json_file
        self.p_dict = read_params(json_file)

    @property
    def is_WT2(self):
        return self.p_dict['analysis_type'].endswith('WT2')

    @property
    def is_one_worm(self):
        analysis_type = self.p_dict['analysis_type']
        return analysis_type.endswith('WT2') or analysis_type.endswith('SINGLE')

    @property
    def nn_filter_to_use(self):
        _nn_filter_to_use = self.p_dict['nn_filter_to_use']
        if not _nn_filter_to_use and 'AEX' in self.p_dict['analysis_type']:
            warnings.warn('setting nn_filter_to_use to "tensorflow_default"'
                          + ' since the analysis type contains AEX.')
            _nn_filter_to_use = "tensorflow_default"

        return _nn_filter_to_use


def parse_splitFOV_params(json_file=''):
    print(f'loading json {json_file} for split fov')

    # if no json file is passed, return empty dict
    if not json_file:
        input_param = {}
    else:  # otherwise laod json file and fill with default values in case
        assert os.path.exists(json_file)
        input_param = {
            "MWP_well_shape": "square",
            "MWP_well_size_mm": 8,
            "MWP_well_masked_edge": 0.1,
            "MWP_camera2channelrig": None,
            "MWP_channel2well": None,
        }
        # load params from file
        with open(json_file) as fid:
            param_in_file = json.load(fid)
        # and override the defaults
        for key in param_in_file:
            if key in input_param:
                input_param[key] = param_in_file[key]
            else:
                raise ValueError(
                    f'Parameter "{key}" is not a valid parameter'
                    f'Change its value in file "{json_file}"'
                    )

    return input_param


class SplitFOVParams:

    def __init__(self, json_file=''):
        if json_file and (json_file in DFLT_SPLITFOV_PARAMS_FILES):
            json_file = os.path.join(DFLT_SPLITFOV_PARAMS_PATH, json_file)

        self.json_file = json_file
        self.p_dict = parse_splitFOV_params(json_file)

    def get_common_params(self):
        """
        get_common_params Return the common parameters in the json file,
            i.e. the ones that do not depend on the filename of the video

        Returns
        -------
        well_shape : str
            shape of wells. square or circle
        well_masked_edge : float
            percentage or a well size to set to black in the compression algo
        well_size_mm : float
            how many mm across is a well? diameter for circular wells,
            length of a side for squared ones
        """
        well_shape = self.p_dict['MWP_well_shape']
        well_masked_edge = self.p_dict['MWP_well_masked_edge']
        well_size_mm = self.p_dict['MWP_well_size_mm']

        return well_shape, well_masked_edge, well_size_mm

    def get_params_from_filename(self, fname):
        """
        Parse filename, try to match it to one of the keys in
        self.p_dict["MWP_camera2channelrig"].
        Check how many matches, return the unique id
        (camera serial, for example), rig name, and channel
        if univocal match, raise Exception if no matches or multiple matches
        """

        # loop through the mapping dict to find an uid in the file name
        matches = {}
        for key, values in self.p_dict["MWP_camera2channelrig"].items():
            if key in str(fname):
                matches[key] = values

        # checks on matches
        n_matches = len(list(matches.keys()))
        if n_matches < 1:
            raise Exception('No camera identifier found in the filename')
        elif n_matches > 1:
            raise Exception(
                f'{fname} matches multiple identifiers: {list(matches.keys())}'
                '. Aborting.'
            )

        # now we know there was only one match, so
        for uid, values in matches.items():
            channel, rig = values

        # and also return the mapping
        mwp_map = np.array(self.p_dict["MWP_channel2well"][channel])

        return uid, rig, channel, mwp_map




if __name__=='__main__':
    json_file = ''
    params = TrackerParams(json_file)


