# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14:33:18 2024

@author: weheliye
"""

import numpy as np
import tables


class readVideoHDF5:
    def __init__(self, fileName, full_img_period=np.inf, dataset="/mask"):
        # to be used when added to the plugin
        self.vid_frame_pos = []
        self.vid_time_pos = []

        try:
            self.fid = tables.File(fileName, "r")
            self.dataset = self.fid.get_node(dataset)
        except IOError:
            raise OSError

        self.tot_frames = self.dataset.shape[0]

        self.width = self.dataset.shape[2]
        self.height = self.dataset.shape[1]
        self.dtype = self.dataset.dtype

        self.tot_pix = self.height * self.width

        # initialize pointer for frames
        self.curr_frame = -1

        # how often we get a full frame
        self.full_img_period = full_img_period
        # self.value2replace = 0

    def read(self):
        self.curr_frame += 1
        if self.curr_frame % self.full_img_period == 0:
            self.full_img = self.dataset[self.curr_frame, :, :]
            # self.value2replace = np.max(self.full_img)
            self.value2replace = np.percentile(self.full_img[self.full_img != 0], 95)

        if self.curr_frame > self.tot_frames:
            return False, None

        image = self.dataset[self.curr_frame, :, :]
        mask_bw = image == 0
        image[mask_bw] = self.value2replace
        return True, image

    def read_frame(self, frame_number):
        # set current frame with -1 offset because it's += 1 in self.read()
        self.curr_frame = frame_number - 1

        return self.read()

    def __len__(self):
        return self.tot_frames

    def release(self):
        # close the buffer
        self.fid.close()


class readFullDataFromVideoHDF5(readVideoHDF5):
    def __init__(self, fileName):
        super().__init__(fileName, dataset="/full_data")

    def read(self):
        self.curr_frame += 1
        if self.curr_frame < self.tot_frames:
            image = self.dataset[self.curr_frame, :, :]
            return (1, image)
        else:
            return (0, [])
