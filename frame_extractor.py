"""
@Author: Noah Chavannes
@Date: 2022-07-12
"""

import os
import subprocess
import shutil
import pprint
import argparse
import math
import logging
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import h5py as h5py
from pathlib import Path
from distutils.util import strtobool


class FramesExtractor:

    def __init__(self, grimace_box_h5_path, mouse_top_h5_path, mouse_front_h5_path, video_top_path, video_front_path,
                 config, file_prefix):
        """
        Init of FramesExtractor. Helper class to extract frames of a video and extract statistics.
        :param grimace_box_h5_path: Path to the SLEAP grimace box analysis h5 file
        :param mouse_top_h5_path: Path to the SLEAP top mouse analysis h5 file
        :param mouse_front_h5_path: Path to the SLEAP front mouse analysis h5 file
        :param video_top_path: Path to the top video
        :param video_front_path: Path to the front video
        :param config: Dictionary containing configuration (parsed arguments=
        :param file_prefix: Prefix that is used for the saved images and the mapping
        """

        self.config = config
        self.file_prefix = file_prefix
        self.grimace_box_h5_filename = grimace_box_h5_path
        self.grimace_box_h5 = h5py.File(grimace_box_h5_path)
        self.mouse_top_h5_filename = mouse_top_h5_path
        self.mouse_top_h5 = h5py.File(mouse_top_h5_path)
        # TODO: Remove when front available
        if mouse_front_h5_path:
            self.mouse_front_h5 = h5py.File(mouse_front_h5_path)
        self.video_top_filename = video_top_path
        # Convert top video to VideoCapture
        self.video_top = cv2.VideoCapture(video_top_path)
        self.video_front_filename = video_front_path
        # Convert front video to VideoCapture
        self.video_front = cv2.VideoCapture(video_front_path)
        # Read the coordinates and likelihoods of different points per frame and store them in a dictionary
        self.frame_values_dict = self.extract_points_from_h5()
        # Calculate the ratio to convert pixels into distance (in millimeters)
        self.px_mm_ratio = self.calc_px_mm_ratio()
        # Get total frame count
        self.total_frame_count = self.get_total_frame_count()
        # Load the rating and mapping dataframe or create them if not existing yet
        self.rating_df = self.load_df('rating.csv')
        self.mapping_df = self.load_df('mapping.csv')

    def extract(self):
        """
        Extracts the best frames from the video
        """

        # Calculate metrics for each frame (e.g. distances from ears to the front of the box, etc.)
        self.calculate_metrics()

        # Calculate the valid frames according to the parameters specified and group then, such that frames
        # within the defined minimal frame distance are in the same group
        frame_groups = self.group_frames(self.calculate_valid_frames())

        # Define list for best frames
        best_frames = []
        # Define variable for index of frame selected in the previous group
        previous_group_frame = None
        # For each frame group, find the best frame
        for group in frame_groups:
            # Find the best frame tuple (frame_index, score)
            res = self.find_best_of_group(group, previous_group_frame)
            # Check if an actual frame was determined
            if res[0] is not None:
                # Add frame to result list
                best_frames.append(res)
                # Set previous group frame
                previous_group_frame = res[0]

        # Sort the best frames according to their score
        best_frames.sort(key=lambda y: y[1], reverse=True)

        # Check if an additional frame limit is enforced
        if self.config['limit_extracted_frames'] > -1:
            # Select the n best frames where n = limit_extracted_frames
            best_frames = best_frames[:self.config['limit_extracted_frames']]

        # For each best frame tuple (frame_index, score)
        for frame in best_frames:
            # Define the result image name
            img_name = f'{self.file_prefix}_{frame[0]}.png'
            # Insert a new empty row in the rating dataframe only containing the image name
            self.rating_df.loc[len(self.rating_df)] = [img_name, '', '', '', '', '', '']
            # Insert a row in the mapping dataframe which maps the image name to the videos and h5 files associated
            self.mapping_df.loc[len(self.mapping_df)] = [self.file_prefix, img_name, frame[0],
                                                         self.video_front_filename, self.video_top_filename,
                                                         self.grimace_box_h5_filename, self.mouse_top_h5_filename]
            # Extract the frontal frame
            extracted_frame_f = self.extract_frame(frame[0], 'f')
            # Save the frontal frame in the output directory
            cv2.imwrite(os.path.join(self.config['output_dir'], img_name), extracted_frame_f)

            # If debugging plots are enabled
            if self.config['debug_plots']:
                # Get frame index
                i = frame[0]
                # Plot annotated top view and extracted front view with corresponding coordinates
                self.debug_plot(i, (self.frame_values_dict['fl_x'][i], self.frame_values_dict['fl_y'][i]),
                                (self.frame_values_dict['fr_x'][i], self.frame_values_dict['fr_y'][i]),
                                (self.frame_values_dict['el_x'][i], self.frame_values_dict['el_y'][i]),
                                (self.frame_values_dict['er_x'][i], self.frame_values_dict['er_y'][i]),
                                (self.frame_values_dict['n_x'][i], self.frame_values_dict['n_y'][i]))

        # Save rating dataframe in output directory
        self.rating_df.to_csv(os.path.join(self.config['output_dir'], 'rating.csv'))
        # Save mapping dataframe in output directory
        self.mapping_df.to_csv(os.path.join(self.config['output_dir'], 'mapping.csv'))

        logging.info(f'Extracted {len(best_frames)} frames from {self.video_front_filename}')

    def statistics(self):
        """
        Calculates the statistics of an extraction run. Can either be run separately or in combination
        with the frame extraction.
        :return: Tuple( number of extracted frames, NDArray(parallel deviation), NDArray(Mean ear to box distance),
                        NDArray(Nose to box distance), NDArray(Likelihood ear left), NDArray(Likelihood ear right),
                        NDArray(Likelihood nose))
        """

        # Calculate metrics for each frame (e.g. distances from ears to the front of the box, etc.)
        self.calculate_metrics()
        # Check if output directory for statistics exists, otherwise create
        FramesExtractor.check_dir(os.path.join(self.config['output_dir'], 'statistics'))

        # Print the chosen parameters
        logging.info(f'---------- Statistics {self.file_prefix} -----------')
        logging.info(f'Current parameters:')
        logging.info(f'norm_dist_mm: {self.config["norm_dist_mm"]}')
        logging.info(f'max_allowed_parallel_deviation: {self.config["max_allowed_parallel_deviation"]}')
        logging.info(f'max_mean_ear_to_box_distance: {self.config["max_mean_ear_to_box_distance"]}')
        logging.info(f'max_nose_box_dist: {self.config["max_nose_box_dist"]}')
        logging.info(f'min_pt_predict_likelihood: {self.config["min_pt_predict_likelihood"]}')
        logging.info(f'min_frame_dist: {self.config["min_frame_dist"]}')
        logging.info(f'limit_extracted_frames: {self.config["limit_extracted_frames"]}')
        # Calculate valid frames
        valid_frames = self.calculate_valid_frames()

        logging.info('-----')

        # Get parallel deviation data
        v = self.frame_values_dict['parallel_deviation']
        logging.info(f'Deviation of parallelism between Line(EarL-EarR) and front of the grimace box (=Line(FR-FL)): '
                     f'{np.nanmean(v):.3f} +- {np.nanstd(v):.3f} (min: {np.nanmin(v):.3f}, max: {np.nanmax(v):.3f}) '
                     f'-> {(v <= self.config["max_allowed_parallel_deviation"]).sum()}/'
                     f'{self.total_frame_count} frames valid (max_allowed_parallel_deviation='
                     f'{self.config["max_allowed_parallel_deviation"]})')
        # Remove NaN values
        dp = v[~np.isnan(v)]
        # Create histogram
        self.create_hist(dp, 'Deviation of parallelism', 'Deviation, 0=parallel, 1=perpendicular', 'parallelism')
        logging.info('-----')

        # Get mean ear to box distance data
        v = self.frame_values_dict['mean_ear_to_box_distance']
        logging.info(f'Mean distance in millimeters from EarL and EarR to front of the grimace box (=Line(FR-FL)): '
                     f'{np.nanmean(v):.3f} +- {np.nanstd(v):.3f} (min: {np.nanmin(v):.3f}, max: {np.nanmax(v):.3f})'
                     f' -> {(v <= self.config["max_mean_ear_to_box_distance"]).sum()}/'
                     f'{self.total_frame_count} frames valid (max_mean_ear_to_box_distance='
                     f'{self.config["max_mean_ear_to_box_distance"]})')
        # Remove NaN values
        me = v[~np.isnan(v)]
        # Create histogram
        self.create_hist(me, 'Mean distance ears to front grimace box', 'Distance [mm]', 'dist_ear_box')
        logging.info('-----')

        # Get nose to box distance data
        v = self.frame_values_dict['nose_box_dist']
        logging.info(f'Distance in millimeters from nose to front of the grimace box (=Line(FR-FL)): '
                     f'{np.nanmean(v):.3f} +- {np.nanstd(v):.3f} (min: {np.nanmin(v):.3f}, max: {np.nanmax(v):.3f})'
                     f' -> {(v <= self.config["max_nose_box_dist"]).sum()}/'
                     f'{self.total_frame_count} frames valid (max_nose_box_dist='
                     f'{self.config["max_nose_box_dist"]})')
        # Remove NaN values
        nd = v[~np.isnan(v)]
        # Create histogram
        self.create_hist(nd, 'Nose to front grimace box distance', 'Distance [mm]', 'dist_nose_box')
        logging.info('-----')

        logging.info('Point Likelihoods:')
        # Get ear left likelihood data
        v = self.frame_values_dict['el_lik']
        # Remove NaN values
        lel = v[~np.isnan(v)]
        # Create histogram
        self.create_hist(nd, 'Likelihood of Ear Left', 'Likelihood', 'likelihood_earl')
        logging.info(f'EarL: {np.nanmean(v):.3f} +- {np.nanstd(v):.3f} (min: {np.nanmin(v):.3f}, max: '
                     f'{np.nanmax(v):.3f}) -> {(v >= self.config["min_pt_predict_likelihood"]).sum()}/'
                     f'{self.total_frame_count} frames valid (min_pt_predict_likelihood='
                     f'{self.config["min_pt_predict_likelihood"]})')

        # Get ear right likelihood data
        v = self.frame_values_dict['er_lik']
        # Remove NaN values
        ler = v[~np.isnan(v)]
        # Create histogram
        self.create_hist(nd, 'Likelihood of Ear Right', 'Likelihood', 'likelihood_earr')
        logging.info(f'EarR: {np.nanmean(v):.3f} +- {np.nanstd(v):.3f} (min: {np.nanmin(v):.3f}, max: '
                     f'{np.nanmax(v):.3f}) -> {(v >= self.config["min_pt_predict_likelihood"]).sum()}/'
                     f'{self.total_frame_count} frames valid (min_pt_predict_likelihood='
                     f'{self.config["min_pt_predict_likelihood"]})')

        # Get nose likelihood data
        v = self.frame_values_dict['n_lik']
        # Remove NaN values
        ln = v[~np.isnan(v)]
        # Create histogram
        self.create_hist(nd, 'Likelihood of Nose', 'Likelihood', 'likelihood_nose')
        logging.info(f'Nose: {np.nanmean(v):.3f} +- {np.nanstd(v):.3f} (min: {np.nanmin(v):.3f}, max: '
                     f'{np.nanmax(v):.3f}) -> {(v >= self.config["min_pt_predict_likelihood"]).sum()}/'
                     f'{self.total_frame_count} frames valid (min_pt_predict_likelihood='
                     f'{self.config["min_pt_predict_likelihood"]})')
        logging.info('=========================================')

        # Define resulting best frame list
        best_frames = []
        # Define variable for index of frame selected in the previous group
        previous_group_frame = None
        # For each frame group, find the best frame
        for group in self.group_frames(self.calculate_valid_frames()):
            # Find the best frame tuple (frame_index, score)
            res = self.find_best_of_group(group, previous_group_frame)
            # Check if an actual frame was determined
            if res[0] is not None:
                # Add frame to result list
                best_frames.append(res)
                # Set previous group frame
                previous_group_frame = res[0]

        logging.info(f'Current parameters yield {len(valid_frames)}/{self.total_frame_count} valid frames.')
        logging.info(f'With consideration of min_frame_dist: {self.config["min_frame_dist"]}, '
                     f'and limit_extracted_frames: {self.config["limit_extracted_frames"]}, '
                     f'{len(best_frames)}/{self.total_frame_count} frames are selected.')
        logging.info('=========================================')

        return len(best_frames), dp, me, nd, lel, ler, ln

    def calculate_metrics(self):
        """
        Calculate different metrics for all frames and store it in frame_values_dict of the class instance:
        - Distance from left ear to the front of the box
        - Distance from right ear to the front of the box
        - Distance from nose to the front of the box
        - Deviation of parallelism between the line connecting the ears of the mouse and the front of the box
        - Mean ear to box distance
        """

        # Calculate the distance from the left ear to the front of the box (=Line(FL,FR))
        ear_l_box_dist = self.calculate_point_line_distance(self.frame_values_dict['el_x'],
                                                            self.frame_values_dict['el_y'],
                                                            self.frame_values_dict['fr_x'],
                                                            self.frame_values_dict['fr_y'],
                                                            self.frame_values_dict['fl_x'],
                                                            self.frame_values_dict['fl_y'])
        # Calculate the distance from the right ear to the front of the box (=Line(FL,FR))
        ear_r_box_dist = self.calculate_point_line_distance(self.frame_values_dict['er_x'],
                                                            self.frame_values_dict['er_y'],
                                                            self.frame_values_dict['fr_x'],
                                                            self.frame_values_dict['fr_y'],
                                                            self.frame_values_dict['fl_x'],
                                                            self.frame_values_dict['fl_y'])
        # Calculate the distance from the nose to the front of the box (=Line(FL,FR))
        nose_box_dist = self.calculate_point_line_distance(self.frame_values_dict['n_x'], self.frame_values_dict['n_y'],
                                                           self.frame_values_dict['fr_x'],
                                                           self.frame_values_dict['fr_y'],
                                                           self.frame_values_dict['fl_x'],
                                                           self.frame_values_dict['fl_y'])
        # Calculate the deviation of parallelism between the line connecting the ears of the mouse
        # and the front of the box.
        parallel_deviation = abs(ear_l_box_dist - ear_r_box_dist) / (
            self.calc_length_of_line(self.frame_values_dict['el_x'], self.frame_values_dict['el_y'],
                                     self.frame_values_dict['er_x'], self.frame_values_dict['er_y']))
        # Calculate the mean distance of the ears to the front of the box.
        mean_ear_to_box_distance = abs(ear_l_box_dist + ear_r_box_dist) / 2

        # Save in frame_values_dict
        self.frame_values_dict['ear_l_box_dist'] = ear_l_box_dist
        self.frame_values_dict['ear_r_box_dist'] = ear_r_box_dist
        self.frame_values_dict['nose_box_dist'] = nose_box_dist
        self.frame_values_dict['parallel_deviation'] = parallel_deviation
        self.frame_values_dict['mean_ear_to_box_distance'] = mean_ear_to_box_distance

    def calculate_valid_frames(self):
        """
        Checks which frames are valid according to the defined parameters
        :return: List of valid frames
        """

        # Define result list
        valid_frames = []

        # For each frame, check conditions
        for i in range(self.total_frame_count):
            # Check conditions according to the defined parameters
            if (self.frame_values_dict['parallel_deviation'][i] <= self.config['max_allowed_parallel_deviation']
                    and self.frame_values_dict['mean_ear_to_box_distance'][i] <= self.config['max_mean_ear_to_box_distance']
                    and self.frame_values_dict['nose_box_dist'][i] <= self.config['max_nose_box_dist']
                    and self.frame_values_dict['er_lik'][i] >= self.config['min_pt_predict_likelihood']
                    and self.frame_values_dict['el_lik'][i] >= self.config['min_pt_predict_likelihood']
                    and self.frame_values_dict['n_lik'][i] >= self.config['min_pt_predict_likelihood']):
                # If conditions are met, add to list of valid frames
                valid_frames.append(i)
        # Return list of valid frames
        return valid_frames

    def calculate_point_line_distance(self, x_p, y_p, x1, y1, x2, y2):
        """
        Calculate the distance (in millimeters) from a point(x_p, y_p) to a line((x1, y1),(x2, y2))
        :param x_p: x coordinate of point
        :param y_p: y coordinate of point
        :param x1: x coordinate of first point of the line
        :param y1: y coordinate of first point of the line
        :param x2: x coordinate of second point of the line
        :param y2: y coordinate of second point of the line
        :return: distance from point to line in millimeters
        """

        dist = abs((y2 - y1) * x_p - (x2 - x1) * y_p + x2 * y1 - y2 * x1)
        dist /= ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
        return dist * self.px_mm_ratio

    @staticmethod
    def calc_length_of_line_px(x1, y1, x2, y2):
        """
        calculate the length of a line in pixels
        :param x1: x coordinate of first point of the line
        :param y1: y coordinate of first point of the line
        :param x2: x coordinate of second point of the line
        :param y2: y coordinate of second point of the line
        :return: length of the line in pixels
        """

        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def calc_length_of_line(self, x1, y1, x2, y2):
        """
        calculate the length of a line in millimeters
        :param x1: x coordinate of first point of the line
        :param y1: y coordinate of first point of the line
        :param x2: x coordinate of second point of the line
        :param y2: y coordinate of second point of the line
        :return: length of the line in millimeters
        """

        return (np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)) * self.px_mm_ratio

    def extract_points_from_h5(self):
        """
        Extracts the point coordinates and likelihoods from the h5 files
        :return: dictionary containing the point data
        """

        # Define result dictionary
        res = {}

        # Extract coordinates and likelihood from front left corner of the grimace box
        res['fl_x'] = self.grimace_box_h5['tracks'][0][0][0]
        res['fl_y'] = self.grimace_box_h5['tracks'][0][1][0]
        res['fl_lik'] = self.grimace_box_h5['point_scores'][0][0]

        # Extract coordinates and likelihood from front right corner of the grimace box
        res['fr_x'] = self.grimace_box_h5['tracks'][0][0][1]
        res['fr_y'] = self.grimace_box_h5['tracks'][0][1][1]
        res['fr_lik'] = self.grimace_box_h5['point_scores'][0][1]

        # Extract coordinates and likelihood from the nose of the mouse
        res['n_x'] = self.mouse_top_h5['tracks'][0][0][3]
        res['n_y'] = self.mouse_top_h5['tracks'][0][1][3]
        res['n_lik'] = self.mouse_top_h5['point_scores'][0][3]

        # Extract coordinates and likelihood from the left ear of the mouse
        res['el_x'] = self.mouse_top_h5['tracks'][0][0][5]
        res['el_y'] = self.mouse_top_h5['tracks'][0][1][5]
        res['el_lik'] = self.mouse_top_h5['point_scores'][0][5]

        # Extract coordinates and likelihood from the right ear of the mouse
        res['er_x'] = self.mouse_top_h5['tracks'][0][0][4]
        res['er_y'] = self.mouse_top_h5['tracks'][0][1][4]
        res['er_lik'] = self.mouse_top_h5['point_scores'][0][4]

        return res

    def calc_px_mm_ratio(self):
        """
        Calculates the ratio to convert pixel distances into millimeters using the provided norm_dist_mm parameter
        :return: conversion ration px->mm
        """

        # Calculate the length of the front of the box in pixels
        length = FramesExtractor.calc_length_of_line_px(self.frame_values_dict['fl_x'].mean(),
                                                        self.frame_values_dict['fr_x'].mean(),
                                                        self.frame_values_dict['fl_y'].mean(),
                                                        self.frame_values_dict['fr_y'].mean())
        # Divide the provided norm distance by the length to get the conversion ratio
        return self.config['norm_dist_mm'] / length

    def extract_frame(self, frame_idx, v='t'):
        """
        Extract a frame from a video
        :param frame_idx: Index of the frame to extract
        :param v: Which video to use (t=top, f=front)
        :return: Extracted frame
        """

        # If top video
        if v == 't':
            # Set video to specified frame
            self.video_top.set(1, frame_idx - 1)
            # Extract frame
            ret, frame = self.video_top.read()
        # If front video
        else:
            # Set video to specified frame
            self.video_front.set(1, frame_idx - 1)
            # Extract frame
            ret, frame = self.video_front.read()
        # Return frame
        return frame

    def group_frames(self, frames):
        """
        Group the valid frames, such that frames within a group are at most min_frame_dist apart
        :param frames: list of valid frames
        :return: list of groups that are within min_frame_dist
        """

        # Define result list
        result = []
        # Define running sublist
        current_sublist = []
        # Init current upper bound
        current_upper_bound = None

        # For each frame in the frame
        for i in frames:
            # If current upper bound is not set
            if current_upper_bound is None:
                # Set current upper bound to frame index + min_frame_dist
                current_upper_bound = i + self.config['min_frame_dist']
            # If current frame index is greater or equal to current upper bound
            if i >= current_upper_bound:
                # append current sublist (=group) to the result
                result.append(current_sublist)
                # reset current sublist
                current_sublist = []
                # reset current upper bound to frame index + min_frame_dist
                current_upper_bound = i + self.config['min_frame_dist']
            # If current frame is below defined upper bound
            if i < current_upper_bound:
                # Add frame to current sublist
                current_sublist.append(i)
        # Add final sublist to result
        result.append(current_sublist)

        # Return list of groups
        return result

    def find_best_of_group(self, frame_group, previous_group_frame):
        """
        Finds best frame of a group
        :param frame_group: list of frames to find best frame in
        :param previous_group_frame: index of the best frame of the previous group
        :return: Tuple (frame index, score) of best frame
        """

        # Define current best frame and current best score
        current_best = None
        current_best_score = -math.inf

        # For each frame in the frame group
        for frame in frame_group:
            # Initialize starting score
            score = 1
            # Subtract parallel deviation from score
            score -= self.frame_values_dict['parallel_deviation'][frame]

            # Calculate ratio of the distance from nose to left edge of the box and from nose to right edge of the box.
            # Subtract this distance from 1 -> if both distances are similar, resulting term will be low. Divide by 2
            # for weighting with other metrics
            norm_nose_to_edges = abs(1 - (self.calc_length_of_line(
                self.frame_values_dict['n_x'][frame],
                self.frame_values_dict['n_y'][frame],
                self.frame_values_dict['fl_x'][frame],
                self.frame_values_dict['fl_y'][frame]) / self.calc_length_of_line(
                self.frame_values_dict['n_x'][frame],
                self.frame_values_dict['n_y'][frame],
                self.frame_values_dict['fr_x'][frame],
                self.frame_values_dict['fr_y'][frame]))) / 2
            # Subtract calculated nose to box metric from remaining score
            score -= norm_nose_to_edges

            # Calculate ratio of the distance from nose to left ear and from nose to right ear.
            # Subtract this distance from 1 -> if both distances are similar, resulting term will be low. Divide by 2
            # for weighting with other metrics
            norm_nose_to_ears = abs(1 - (self.calc_length_of_line(
                self.frame_values_dict['n_x'][frame],
                self.frame_values_dict['n_y'][frame],
                self.frame_values_dict['el_x'][frame],
                self.frame_values_dict['el_y'][frame]) / self.calc_length_of_line(
                self.frame_values_dict['n_x'][frame],
                self.frame_values_dict['n_y'][frame],
                self.frame_values_dict['er_x'][frame],
                self.frame_values_dict['er_y'][frame]))) / 2
            # Subtract calculated nose to ear metric from remaining score
            score -= norm_nose_to_ears

            # TODO: Maybe define a rang withing the mice's noses and ears have to be instead of just using distance
            # Subtract the nose to front of box distance (divided by 200 for weighting). The closer the mouse to the box
            # the higher the remaining score
            score -= self.frame_values_dict['nose_box_dist'][frame] / 200

            # Calculate the average likelihood of the nose and the ear coordinates
            avg_lik = (self.frame_values_dict['n_lik'][frame] + self.frame_values_dict['el_lik'][frame] +
                       self.frame_values_dict['er_lik'][frame]) / 3

            # Subtract average likelihood from 1 (higher likelihood = less deduction) and divide by 2 for weighting.
            # Subtract from remaining score
            score -= (1 - avg_lik) / 2

            # If the score of the current frame is better than the previous best score
            # AND if the frame is at least min_frame_dist away from the index of the best frame of the previous group
            if (score > current_best_score and
                    (previous_group_frame is None or
                     (previous_group_frame is not None and
                      (previous_group_frame + self.config['min_frame_dist']) < frame))):
                # Save score of current frame as best score
                current_best_score = score
                # Save index of current frame as current best frame
                current_best = frame

        # Return tuple (frame index, frame score) of best frame of group
        return current_best, current_best_score

    def debug_plot(self, frame_idx, fl, fr, el, er, n):
        """
        Creates a debug plot showing the top view annotated with coordinates and the corresponding front view
        :param frame_idx: index of the frame
        :param fl: Front left of grimace box: Tuple (x,y)
        :param fr: Front right of grimace box: Tuple (x,y)
        :param el: Left ear of mouse: Tuple (x,y)
        :param er: Right ear of mouse: Tuple (x,y)
        :param n: Nose ear of mouse: Tuple (x,y)
        """

        # Clear figure
        plt.clf()
        # Get current figure
        fig = plt.gcf()
        # Set size
        fig.set_size_inches(8, 3)

        # Define first subplot
        plt.subplot(1, 2, 1)
        # Extract and plot top image
        img = self.extract_frame(frame_idx, "t")
        plt.imshow(img)
        # Define coordinates of the line of the front of the grimace box
        x = [fl[0], fr[0]]
        y = [fl[1], fr[1]]
        # Plot the front line of the grimace box
        plt.plot(x, y, marker="o", markersize=2, color="red")
        # Define coordinates of the ears of the mouse
        x = [el[0], er[0]]
        y = [el[1], er[1]]
        # Plot the line between the ears of the mouse
        plt.plot(x, y, marker="o", markersize=2, color="green")
        # Plot the nose
        plt.plot(n[0], n[1], marker="o", markersize=2, color="blue")
        # Set title
        plt.title(f'{frame_idx} - TOP')

        # Define the second subplot
        plt.subplot(1, 2, 2)
        # Extract and plot front image
        img = self.extract_frame(frame_idx, "f")
        plt.imshow(img)

        # Adjust spacing
        plt.subplots_adjust(left=0.1, top=0.99, right=0.9, bottom=0.01, hspace=0, wspace=0.2)
        # Set title
        plt.title(f'{frame_idx} - FRONT')
        # Check if debug directory exists or create
        FramesExtractor.check_dir(os.path.join(self.config['output_dir'], 'debug'))
        # Save plot
        plt.savefig(os.path.join(self.config['output_dir'], 'debug', f'{self.file_prefix}_{frame_idx}.png'))
        # Enable tho show plots one by one
        # plt.show()

    @staticmethod
    def check_dir(path):
        """
        Check if directory exists or create
        :param path: Path to directory
        """
        Path(path).mkdir(parents=True, exist_ok=True)

    def create_hist(self, data, title, x_label, file_suffix):
        """
        Create histogram
        :param data: Data for histogram
        :param title: Title
        :param x_label: X-Label
        :param file_suffix: Suffix for filename
        """
        # Clear figure
        plt.clf()
        # Create histogram with 20 bins
        plt.hist(data, bins=20)
        # Set title
        plt.title(f'{self.file_prefix}: {title}')
        # Set X-Label
        plt.xlabel(x_label)
        # Set Y-Label
        plt.ylabel('Count')
        # Save plot in statistics directory
        plt.savefig(os.path.join(self.config['output_dir'], 'statistics', f'{self.file_prefix}{file_suffix}.png'))

    def load_df(self, file):
        """
        Load a dataframe or create if it does not exist
        :param file: dataframe path
        :return: Dataframe
        """

        # Define the path of the dataframe
        path = os.path.join(self.config['output_dir'], file)

        # If dataframe exists
        if os.path.exists(path):
            # Read from file and return
            return pd.read_csv(path, index_col=0)
        # If dataframe does not exist
        else:
            # If filename contains mapping
            if 'mapping' in file:
                # Create a new mapping dataframe and return
                return pd.DataFrame(
                    columns=['video_key', 'image_name', 'frame_idx', 'video_front_name', 'video_top_name',
                             'grimace_box_h5_file', 'mouse_top_h5_file'])
            # if file name does not contain mapping (=rating dataframe)
            else:
                # Create a new rating dataframe and return
                return pd.DataFrame(
                    columns=['image_name', 'mgs_orbital_tightening', 'mgs_nose_bulge', 'mgs_cheek_bulge',
                             'mgs_ear_position', 'mgs_whisker_change', 'mgs_not_ratable'])

    def get_total_frame_count(self):
        """
        Returns the total number of frames in the videos
        :return: number of frames
        """

        # Get the total number of frames and return
        return len(self.frame_values_dict['el_x'])

# ----------------------------------------------------------------------------------------------------------------------
#  END OF FramesExtractor   --------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def check_step(config):
    """
    Checks which step should be processed:
    - 1: Convert videos
    - 2: Extract h5 from converted videos using SLEAP
    - 3: Extract frames form converted videos and h5 files
    - (-1): Nothing to process
    :param config: parsed configuration
    :return: process step
    """

    # If number of videos to convert is greater than 0
    if len(get_videos_to_convert(config)) > 0:
        # Check that there are matching top and front videos
        check_matching_files(config, 'avi')
        return 1

    # If number of converted videos to extract h files from is greater than 0
    if len(get_videos_to_extract_h5(config)) > 0:
        # Check that there are matching top and front videos
        check_matching_files(config, 'mp4')
        return 2

    # If number of videos in which no frames were extracted is greater than 0
    if len(get_videos_to_extract_frames(config)) > 0:
        # Check that there are matching top and front videos
        check_matching_files(config, 'mp4')
        return 3

    # If nothing to process
    return -1


def check_matching_files(config, file_format):
    """
    Checks if there are matching front and top videos
    :param config: parsed configuration
    :param file_format: file format to check
    """
    # Get the files ending with file_format containing the front_camera_file_pattern
    c_f = Path(config['input_dir']).glob(f'**/*{config["front_camera_file_pattern"]}*.{file_format}')
    # Convert to files
    c_f_files = [x for x in c_f if x.is_file()]
    # Convert to file name only
    c_f_files_name_only = [x.name for x in c_f_files]
    # Get the files ending with file_format containing the top_camera_file_pattern
    c_t = Path(config['input_dir']).glob(f'**/*{config["top_camera_file_pattern"]}*.{file_format}')
    # Convert to files
    c_t_files = [x for x in c_t if x.is_file()]
    # Convert to file name only
    c_t_files_name_only = [x.name for x in c_t_files]

    # For each file name of the front camera files
    for file in c_f_files_name_only:
        # Create filename with replacing front_camera_file_pattern with top_camera_file_pattern
        file_name_to_check = file.replace(config["front_camera_file_pattern"], config["top_camera_file_pattern"])
        # Check that filename exists in the list of top camera files
        if file_name_to_check not in c_t_files_name_only:
            # Raise exception if it does not exist
            raise Exception(f"No matching top video found for {file}")

    # For each file name of the top camera files
    for file in c_t_files_name_only:
        # Create filename with replacing top_camera_file_pattern with front_camera_file_pattern
        file_name_to_check = file.replace(config["top_camera_file_pattern"], config["front_camera_file_pattern"])
        # Check that filename exists in the list of front camera files
        if file_name_to_check not in c_f_files_name_only:
            # Raise exception if it does not exist
            raise Exception(f"No matching front video found for {file}")


def get_videos_to_convert(config):
    """
    Checks which videos still need to be converted
    :param config: parsed configuration
    :return: list of videos to be converted
    """

    # Search input directory for files
    v_f = Path(config['input_dir']).glob(f'**/*')
    # Convert result to files
    v_files = [x for x in v_f if x.is_file()]
    # Create sublist of *.avi files
    avi = [x for x in v_files if x.name[-4:] == '.avi']
    # Create sublist of *.mp4 files without the '_c' suffix
    mp4_c = [x.name[:-6] for x in v_files if x.name[-4:] == '.mp4']

    # Define list of videos to be converted
    videos_to_convert = []
    # For each avi video
    for v in avi:
        # Check if name of video without '.avi' is not present in the list on converted mp4 files
        if v.name[:-4] not in mp4_c:
            # If not present, append to list to be converted
            videos_to_convert.append(v)

    # Return list of videos still to be converted
    return videos_to_convert


def get_videos_to_extract_h5(config):
    """
    Checks which videos still need to have their h5 files extracted
    :param config: parsed configuration
    :return: list of videos to have their h5 files extracted
    """

    # Search input directory for files
    v_f = Path(config['input_dir']).glob(f'**/*')
    # Convert result to files
    v_files = [x for x in v_f if x.is_file()]
    # Create sublist of *.mp4 files
    mp4 = [x for x in v_files if x.name[-4:] == '.mp4']
    # Create sublist of file names of files ending in .h5
    h5_names = [x.name for x in v_files if x.name[-3:] == '.h5']

    # Define list of videos to have their h5 files extracted
    videos_to_extract = []
    # For each mp4 video
    for v in mp4:
        # If the top_camera_file_pattern is present in the video name
        if config['top_camera_file_pattern'] in v.name:
            # If there are less than 2 .h5 files (Mouse and GrimaceBox) with the video name contained
            if count_occurrence_in_list(v.name, h5_names) < 2:
                # Append to the list of videos still to have h5 data extracted
                videos_to_extract.append(v)

    # Return list of videos to have their h5 files extracted
    return videos_to_extract


def get_videos_to_extract_frames(config):
    """
    Check which videos still need to have their frames extracted
    :param config: parsed configuration
    :return: list of Tuples of videos to have their frames extracted: Tuple(video_prefix, video_file)
    """

    # Search input directory for files
    v_f = Path(config['input_dir']).glob(f'**/*')
    # Convert result to files
    v_files = [x for x in v_f if x.is_file()]
    # Create sublist of *.mp4 files
    mp4 = [x for x in v_files if x.name[-4:] == '.mp4']

    # Define list of videos to have their frames extracted
    videos_to_extract = []
    # If mapping.csv exists
    if os.path.exists(os.path.join(config['output_dir'], 'mapping.csv')):
        # Read the mapping.csv file into a dataframe
        df = pd.read_csv(os.path.join(config['output_dir'], 'mapping.csv'), index_col=0)
        # Find the next video index (by looking at the unique video_key values, removing the 'v' prefix, taking the max
        # value and adding 1)
        idx = max([int(x[1:]) for x in df['video_key'].unique().tolist()]) + 1
        # For each video
        for v in mp4:
            # If top_camera_file_pattern is present in the video name
            if config['top_camera_file_pattern'] in v.name:
                # If the full path of the video is not present in the video_top_name entries of the dataframe (=means it
                # was not processed yet)
                if str(v.resolve()) not in df['video_top_name'].tolist():
                    # Add a tuple to the list of videos to have frames extracted: Tuple(video_prefix, video_file)
                    videos_to_extract.append((idx, v))
                    # Increment the index
                    idx += 1
    # If mapping.csv does not exist
    else:
        # Init the index
        idx = 0
        # For each video
        for v in mp4:
            # If top_camera_file_pattern is present in the video name
            if config['top_camera_file_pattern'] in v.name:
                # Add a tuple to the list of videos to have frames extracted: Tuple(video_prefix, video_file)
                videos_to_extract.append((idx, v))
                # Increment the index
                idx += 1

    # Return list of tuples of videos to have their frames files extracted
    return videos_to_extract


def count_occurrence_in_list(name, name_list):
    """
    Counts the occurences of a file name in a list of file names
    :param name: name of the file
    :param name_list: list of file names
    :return: count of the occurrences
    """

    # Init count
    count = 0

    # For each name in the list
    for n in name_list:
        # If is present
        if name in n:
            # Increase count
            count += 1
    # Return count
    return count


def create_hist(config, data, title, x_label, file_name):
    """
    Create histogram
    :param config: parsed configuration
    :param data: Data for histogram
    :param title: Title
    :param x_label: X-Label
    :param file_name: file name of the plot
    """
    # Clear figure
    plt.clf()
    # Create histogram with 20 bins
    plt.hist(data, bins=20)
    # Set title
    plt.title(title)
    # Set X-Label
    plt.xlabel(x_label)
    # Set Y-Label
    plt.ylabel('Count')
    # Save plot in statistics directory
    plt.savefig(os.path.join(config['output_dir'], 'statistics', file_name))


def convert_videos(config):
    """
    Convert videos from avi to mp4 using ffmpeg
    :param config: parsed configuration
    """

    # For each video in the list of videos to still be converted
    for v in get_videos_to_convert(config):
        # Prepare terminal command for ffmpeg
        cmd = f'ffmpeg -i "{v.resolve()}" -vf scale=928:576 -c:v libx264 -pix_fmt ' \
              f'yuv420p -preset superfast -crf 23 -max_muxing_queue_size 99999 "{str(v.resolve())[:-4]}_c.mp4"'
        # Execute terminal command
        subprocess.call(cmd, shell=True)


def extract_h5(config):
    """
    Extract the h5 files from converted videos using SLEAP
    :param config: parsed configuration
    """

    # For each video in the list of videos to have their h5 data extracted
    for v in get_videos_to_extract_h5(config):

        # Define file name for the mouse .slp file
        slp_file_m = f'{v.resolve()}.MOUSE.slp'
        # Define file name for the mouse .h5 file
        h5_file_m = f'{v.resolve()}.MOUSE.analysis.h5'

        # Check if the .slp or .h5 exists
        if os.path.exists(slp_file_m) or os.path.exists(h5_file_m):
            # If one of the two exists, skip
            logging.info(f'Skipping {slp_file_m} since it already exists or has been converted to h5.')
        # If both files do not exist
        else:
            # Prepare command to extract .slp file form the mouse model
            cmd = f'sleap-track "{v.resolve()}" -o "{slp_file_m}" -m "{config["sleap_model_mouse_path"]}"'
            # Run command
            subprocess.call(cmd, shell=True)

        # Check if the .h5 file exists
        if os.path.exists(h5_file_m):
            # If .h5 file exists, skip
            logging.info(f'Skipping {h5_file_m} since it already exists.')
        # If not (also means that .slp file exists, otherwise it would have created it a few lines above)
        else:
            # Prepare command to convert .slp into .h5
            cmd = f'sleap-convert "{slp_file_m}" --format analysis'
            # Run command
            subprocess.call(cmd, shell=True)

        # Check if the .slp and .h5 exists
        if os.path.exists(slp_file_m) and os.path.exists(h5_file_m):
            # Remove the .slp file for cleanup
            os.remove(slp_file_m)

        # Define file name for the grimace box .slp file
        slp_file_g = f'{v.resolve()}.GrimaceBox.slp'
        # Define file name for the grimace box .h5 file
        h5_file_g = f'{v.resolve()}.GrimaceBox.analysis.h5'

        # Check if the .slp or .h5 exists
        if os.path.exists(slp_file_g) or os.path.exists(h5_file_g):
            # If one of the two exists, skip
            logging.info(f'Skipping {slp_file_g} since it already exists or has been converted to h5.')
        # If both files do not exist
        else:
            # Prepare command to extract .slp file form the grimace box model
            cmd = f'sleap-track "{v.resolve()}" -o "{slp_file_g}" -m "{config["sleap_model_grimace_box_path"]}"'
            # Run command
            subprocess.call(cmd, shell=True)

        # Check if the .h5 file exists
        if os.path.exists(h5_file_g):
            # If .h5 file exists, skip
            logging.info(f'Skipping {h5_file_g} since it already exists.')
        # If not (also means that .slp file exists, otherwise it would have created it a few lines above)
        else:
            # Prepare command to convert .slp into .h5
            cmd = f'sleap-convert "{slp_file_g}" --format analysis'
            # Run command
            subprocess.call(cmd, shell=True)

        # Check if the .slp and .h5 exists
        if os.path.exists(slp_file_g) and os.path.exists(h5_file_g):
            # Remove the .slp file for cleanup
            os.remove(slp_file_g)


def extract_frames(config):
    """
    Extract frames from the videos
    :param config: parsed configuration
    """

    # Define variables to sum up statistics of all videos
    total_frames = 0
    dp, me, nd, lel, ler, ln = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # for each Tuple (video_index, video)
    for i, v in get_videos_to_extract_frames(config):
        # Get the mouse .h5 file
        m_h5 = f'{v.resolve()}.MOUSE.analysis.h5'
        # Check that the file exists
        if not os.path.exists(m_h5):
            # Otherwise raise exception
            raise Exception("H5 File missing for the mouse network")
        # Get the grimace box .h5 file
        g_h5 = f'{v.resolve()}.GrimaceBox.analysis.h5'
        # Check that the file exists
        if not os.path.exists(g_h5):
            # Otherwise raise exception
            raise Exception("H5 File missing for the grimace box network")

        # Define the name of the front facing video by replacing top_camera_file_pattern with front_camera_file_pattern
        front_v = str(v.resolve()).replace(config['top_camera_file_pattern'], config['front_camera_file_pattern'])

        # If frames should be extracted from full resolution front facing camera
        if config['extract_from_original']:
            # Replace '_c.mp4' of the converted video with '.avi'
            front_v = f'{front_v[:-6]}.avi'

        # Define a FramesExtractor
        e = FramesExtractor(grimace_box_h5_path=g_h5,
                            mouse_top_h5_path=m_h5,
                            mouse_front_h5_path='',
                            video_front_path=front_v,
                            video_top_path=str(v.resolve()),
                            config=vars(args),
                            file_prefix=f'v{i}')

        # Extract statistics from the FramesExtractor instance
        total_frames_, dp_, me_, nd_, lel_, ler_, ln_ = e.statistics()

        # Assign extracted values to the variables that track the statistics over all videos
        total_frames += total_frames_
        dp = np.concatenate((dp, dp_))
        me = np.concatenate((me, me_))
        nd = np.concatenate((nd, nd_))
        lel = np.concatenate((lel, lel_))
        ler = np.concatenate((ler, ler_))
        ln = np.concatenate((ln, ln_))

        # If the statistics_only flag is False, frames are extracted
        if not config['statistics_only']:
            # Extract frames from video
            e.extract()

        # Create histograms for combined data of all videos
        create_hist(config, dp, 'All: Deviation of parallelism', 'Deviation, 0=parallel, 1=perpendicular', 'all_parallelism.png')
        create_hist(config, me, 'All: Mean distance ears to front grimace box', 'Distance [mm]', 'all_dist_ear_box.png')
        create_hist(config, nd, 'All: Nose to front grimace box distance', 'Distance [mm]', 'all_dist_nose_box.png')
        create_hist(config, lel, 'All: Likelihood of Ear Left', 'Likelihood', 'all_likelihood_earl.png')
        create_hist(config, ler, 'All: Likelihood of Ear Right', 'Likelihood', 'all_likelihood_earr.png')
        create_hist(config, ln, 'All: Likelihood of Nose', 'Likelihood', 'all_likelihood_nose.png')

    # Print information amount the total amount of frames that will be/were extracted
    logging.info('==================================')
    logging.info(f'In total, {total_frames} frames {"will be" if config["statistics_only"] else "were"} extracted!')
    logging.info('==================================')


# Main Entry Point
if __name__ == '__main__':
    # Set logging level to INFO
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    # Define an argument parser
    parser = argparse.ArgumentParser()
    # Define the arguments
    parser.add_argument('--input_dir', '-i', nargs='?', const=1, type=str, default=os.path.join('.', 'input'),
                        help='Path to input directory, in which the .avi files are stored')
    parser.add_argument('--output_dir', '-o', nargs='?', const=1, type=str, default=os.path.join('.', 'output'),
                        help='Path to output directory directory, in which extracted frames and the rating.csv, '
                             'mapping.csv (and debug plots if enabled) are stored')
    parser.add_argument('--norm_dist_mm', '-nd', nargs='?', const=1, type=float, default=70.0,
                        help='Distance of the front of the grimace box in millimeters')
    parser.add_argument('--max_allowed_parallel_deviation', '-pd', nargs='?', const=1, type=float, default=0.15,
                        help='Max. deviation of the line between the ears of the mouse and the front of the '
                             'grimace box (values from 0.0 - 1.0)')
    parser.add_argument('--max_mean_ear_to_box_distance', '-me', nargs='?', const=1, type=float, default=50.0,
                        help='Max. average distance  between the ears of the mouse and the front of the '
                             'grimace box in millimeters (([ear left to box] + [ear right to box])/2)')
    parser.add_argument('--max_nose_box_dist', '-mn', nargs='?', const=1, type=float, default=20.0,
                        help='Max distance from nose to grimace box in millimeters')
    parser.add_argument('--min_pt_predict_likelihood', '-ml', nargs='?', const=1, type=float, default=0.5,
                        help='Min. likelihood of the predicted points for ears and nose')
    parser.add_argument('--min_frame_dist', '-fd', nargs='?', const=1, type=float, default=10.0,
                        help='Min distance consecutive valid frames have to be apart')
    parser.add_argument('--debug_plots', '-d', nargs='?', type=strtobool, const=True, default=False,
                        help='Enable to see plots showing the extraction of the frames. '
                             'Plots are also saved in the output directory')
    parser.add_argument('--limit_extracted_frames', '-lf', nargs='?', const=1, type=int, default=-1,
                        help='Limit the number of extracted frame per video (Upper limit)')
    parser.add_argument('--statistics_only', '-s', nargs='?', const=True, type=strtobool, default=False,
                        help='Activate to see extraction statistics of current parameters')
    parser.add_argument('--front_camera_file_pattern', '-fc', nargs='?', const=1, type=str, default='c1Grimace',
                        help='File name mask for front camera')
    parser.add_argument('--extract_from_original', '-orig', nargs='?',const=True, type=strtobool, default=False,
                        help='Set to true if images should be extracted from te original frontal video')
    parser.add_argument('--top_camera_file_pattern', '-tc', nargs='?', const=1, type=str, default='c2Grimace',
                        help='File name mask for top camera')
    parser.add_argument('--sleap_model_mouse_path', '-sm', nargs='?', const=1, type=str,
                        default=os.path.join('.', 'mouse_model'),
                        help='Path to the mouse model')
    parser.add_argument('--sleap_model_grimace_box_path', '-sg', nargs='?', const=1, type=str,
                        default=os.path.join('.', 'grimace_box_model'),
                        help='Path to the grimace box model')
    parser.add_argument('--force_clean_output_directory', '-f', nargs='?', const=True, type=strtobool, default=False,
                        help='Cleans the output directory first. Use this flag if you want to rerun the '
                             'extraction of frames or the statistics. Be aware that all content of the '
                             'output older will be deleted first.')

    # Parse the arguments
    args = parser.parse_args()

    # Convert to dictionary
    parsed_config = vars(args)

    # Pretty Print the configuration dictionary
    pprint.pprint(parsed_config)

    # Create the output directory if it does not exist
    Path(parsed_config['output_dir']).mkdir(parents=True, exist_ok=True)

    # If the flag force_clean_output_directory is set
    if parsed_config['force_clean_output_directory']:
        # Promt the user that all data from the output directory will be deleted
        text = input(f"Force cleaning removes all files from the output directory ({parsed_config['output_dir']}). "
                     f"Continue? [y/N]:")

        # If the user answers with "y"
        if text.lower() == 'y':
            # Recursively delete all the files from the output directory
            for root, dirs, files in os.walk(parsed_config['output_dir']):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))
        # If the user answers with anything other than "y"
        else:
            # Abort and exit program
            logging.info('Aborted')
            exit(0)

    # Analyse which step should be executed
    step = check_step(parsed_config)

    # Execute conversion, SLEAP analysis and frame extraction
    if step == 1:
        logging.info('Convert, SLEAP Analysis and Extract Frames')
        convert_videos(parsed_config)
        extract_h5(parsed_config)
        extract_frames(parsed_config)
    # Execute SLEAP analysis and frame extraction
    elif step == 2:
        logging.info('SLEAP Analysis and Extract Frames')
        extract_h5(parsed_config)
        extract_frames(parsed_config)
    # Execute frame extraction
    elif step == 3:
        logging.info('Extract Frames')
        extract_frames(parsed_config)
    # Do nothing
    elif step == -1:
        logging.info('Nothing to process (use --force_clean_output_directory if you want to '
                     'rerun frame extraction or statistics)')
