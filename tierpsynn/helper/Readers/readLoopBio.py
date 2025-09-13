#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 16:56:55 2016

@author: ajaver
"""


class readLoopBio:
    def __init__(self, video_file):
        import imgstore

        self.vid = imgstore.new_for_filename(video_file)

        self.first_frame = self.vid.frame_min
        self.frame_max = self.vid.frame_max
        self.tot_frames = self.frame_max - self.first_frame + 1  # deprecated

        img, (frame_number, frame_timestamp) = self.vid.get_next_image()
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.dtype = img.dtype

        self.vid.close()
        self.vid = imgstore.new_for_filename(video_file)
        self.frames_read = []
        self.fps = self.get_fps_from_metadata()

    def read(self):
        if not self.frames_read or self.frames_read[-1][0] < self.frame_max:
            img, (frame_number, frame_timestamp) = self.vid.get_next_image()
            self.frames_read.append((frame_number, frame_timestamp))
            return True, img
        else:
            return False, None

    def read_frame(self, frame_number):
        frame_to_read = self.first_frame + frame_number
        if frame_to_read >= self.frame_max:
            return False, None

        img, (frame_number, frame_timestamp) = self.vid.get_image(frame_to_read)
        self.frames_read.append((frame_number, frame_timestamp))
        return True, img

    def __len__(self):
        return self.frame_max - self.first_frame + 1

    def release(self):
        return self.vid.close()

    def get_fps_from_metadata(self):
        img, (_, first_timestamp) = self.vid.get_next_image()
        img, (_, second_timestamp) = self.vid.get_next_image()

        # Calculate FPS from the time difference between frames
        time_diff = second_timestamp - first_timestamp
        if time_diff > 0:
            return 1 / time_diff
        else:
            return 1  # Default fallback FPS
