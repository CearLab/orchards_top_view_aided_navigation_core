#!/usr/bin/env python

import datetime
import time
import cv2
import pickle
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped

from computer_vision.contours_scan_cython import contours_scan
from computer_vision import segmentation
from computer_vision import maps_generation


PROFILE_SCAN_GENERATOR = False
TRACK_NAN_IN_SCANS = False
TRACK_INF_IN_SCANS = False

class SyntheticScanGenerator(object):
    def __init__(self):
        rospy.init_node('synthetic_scan_generator')
        self.namespace = rospy.get_namespace()
        self.frame_id = rospy.get_param('~frame_id')
        self.min_angle = rospy.get_param('~min_angle')
        self.max_angle = rospy.get_param('~max_angle')
        self.samples_num = rospy.get_param('~samples_num')
        self.min_distance = rospy.get_param('~min_distance') # TODO: this is in pixels - not so good!!!!!
        self.max_distance = rospy.get_param('~max_distance') # TODO: this is also in pixels - not so good!!!!!
        self.resolution = rospy.get_param('~resolution')
        self.r_primary_search_samples = rospy.get_param('~r_primary_search_samples')
        self.r_secondary_search_step = rospy.get_param('~r_secondary_search_step')
        self.noise_sigma = rospy.get_param('~scan_noise_sigma')
        self.scans_pickle_path = None if rospy.get_param('~scans_pickle_path') == 'None' else rospy.get_param('~scans_pickle_path')
        if self.scans_pickle_path is not None:
            with open(self.scans_pickle_path) as f:
                self.timestamp_to_scan = pickle.load(f)
        localization_image_path = rospy.get_param('~localization_image_path')
        self.localization_image = cv2.cvtColor(cv2.imread(localization_image_path), cv2.COLOR_BGR2GRAY)
        self.prev_scan_time = None
        self.scan_pub = rospy.Publisher('scan', LaserScan, queue_size=1)
        rospy.Subscriber('/ugv_pose', PointStamped, self.virtual_pose_callback)
        self.scan_idx = 0
        if PROFILE_SCAN_GENERATOR:
            self.mean_scan_time = None
        if TRACK_NAN_IN_SCANS:
            self.all_nans = 0
            self.any_nan = 0
        if TRACK_INF_IN_SCANS:
            self.mean_inf_rate = None
        rospy.spin()

    def _publish_scan_message(self, center_x, center_y, timestamp, contours_image):
        if self.prev_scan_time is None:
            self.prev_scan_time = datetime.datetime.now()
            return
        if PROFILE_SCAN_GENERATOR:
            ts = time.time()
        if self.scans_pickle_path is None:
            scan_ranges = contours_scan.generate(contours_image,
                                                 center_x=center_x,
                                                 center_y=center_y,
                                                 min_angle=self.min_angle,
                                                 max_angle=self.max_angle,
                                                 samples_num=self.samples_num,
                                                 min_distance=self.min_distance,
                                                 max_distance=self.max_distance,
                                                 resolution=self.resolution,
                                                 r_primary_search_samples=self.r_primary_search_samples,
                                                 r_secondary_search_step=self.r_secondary_search_step)
        else:
            scan_ranges = self.timestamp_to_scan[timestamp]
        if self.noise_sigma != 0:
            noise = np.random.normal(loc=0, scale=self.noise_sigma, size=len(scan_ranges))
            scan_ranges = scan_ranges + noise
        curr_scan_time = datetime.datetime.now()
        if PROFILE_SCAN_GENERATOR:
            te = time.time()
            delta = (te - ts)
            if self.scan_idx == 0:
                self.mean_scan_time = delta
            else:
                self.mean_scan_time = float(self.mean_scan_time) * (self.scan_idx - 1) / self.scan_idx + delta / self.scan_idx
            rospy.loginfo('%s :: Synthetic scan generation time: %f[sec], mean: %f[sec]' % (self.namespace, delta, self.mean_scan_time))
        if TRACK_NAN_IN_SCANS:
            if np.isnan(scan_ranges).any():
                self.any_nan += 1
            if np.isnan(scan_ranges).all():
                self.all_nans += 1
                return # TODO: ???????????????????
            rospy.loginfo('%s :: Any NaN occurrences: %d' % (self.namespace, self.any_nan))
            rospy.loginfo('%s :: All NaN occurrences: %d' % (self.namespace, self.all_nans))
        if TRACK_INF_IN_SCANS:
            inf_rate = float(np.sum(np.isinf(scan_ranges))) / len(scan_ranges)
            if self.scan_idx == 0:
                self.mean_inf_rate = inf_rate
            else:
                self.mean_inf_rate = float(self.mean_inf_rate) * (self.scan_idx - 1) / self.scan_idx + inf_rate / self.scan_idx
            rospy.loginfo('%s :: Mean inf rate: %f' % (self.namespace, self.mean_inf_rate))
        self.scan_idx += 1
        laser_scan = LaserScan()
        laser_scan.header.stamp = rospy.rostime.Time.now()
        laser_scan.header.frame_id = self.frame_id
        laser_scan.angle_min = self.min_angle
        laser_scan.angle_max = self.max_angle
        laser_scan.angle_increment = (self.max_angle - self.min_angle) / self.samples_num
        laser_scan.scan_time = (curr_scan_time - self.prev_scan_time).seconds
        laser_scan.range_min = self.min_distance * self.resolution
        laser_scan.range_max = self.max_distance * self.resolution
        laser_scan.ranges = scan_ranges
        self.scan_pub.publish(laser_scan)
        self.prev_scan_time = curr_scan_time

    def virtual_pose_callback(self, message):
        self._publish_scan_message(message.point.x, message.point.y, message.header.stamp, self.localization_image)


if __name__ == '__main__':
    SyntheticScanGenerator()