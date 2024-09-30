# # -*- coding: utf-8 -*-
# """
# Created on Tue Jul  7 11:29:01 2015

# @author: ajaver
# """
import os
import sys
import warnings





with warnings.catch_warnings():
    #to remove annoying warnings in case matplotlib was imported before
    warnings.simplefilter("ignore")

try:
    # PyInstaller creates a temp folder and stores path in _MEIPASS
    base_path = sys._MEIPASS
except Exception:
    base_path = os.path.dirname(__file__)

AUX_FILES_DIR = os.path.abspath(os.path.join(base_path, 'extras'))
DFLT_PARAMS_PATH = os.path.join(AUX_FILES_DIR, 'param_files')

DFLT_PARAMS_FILES = sorted(
    [x for x in os.listdir(DFLT_PARAMS_PATH) if x.endswith('.json')])

DFLT_SPLITFOV_PARAMS_PATH = os.path.join(AUX_FILES_DIR, 'splitfov_param_files')
DFLT_SPLITFOV_PARAMS_FILES = sorted(
    [x for x in os.listdir(DFLT_SPLITFOV_PARAMS_PATH) if x.endswith('.json')])



#this will be true if it is a pyinstaller "frozen" binary

