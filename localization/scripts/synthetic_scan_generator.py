#!/usr/bin/env python

import datetime
import time
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from computer_vision.contours_scan_cython import contours_scan
from computer_vision import segmentation
from computer_vision import maps_generation


PROFILE_SCAN_GENERATOR = False
TRACK_NAN_SCANS = False

class SyntheticScanGenerator(object):
    def __init__(self):
        rospy.init_node('synthetic_scan_generator')
        virtual_ugv_mode = True # TODO: remove this logic!!!!!
        self.frame_id = rospy.get_param('~frame_id')
        self.min_angle = rospy.get_param('~min_angle')
        self.max_angle = rospy.get_param('~max_angle')
        self.samples_num = rospy.get_param('~samples_num')
        self.min_distance = rospy.get_param('~min_distance')
        self.max_distance = rospy.get_param('~max_distance')
        self.resolution = rospy.get_param('~resolution')
        self.r_primary_search_samples = rospy.get_param('~r_primary_search_samples')
        self.r_secondary_search_step = rospy.get_param('~r_secondary_search_step')
        self.prev_scan_time = None
        self.scan_pub = rospy.Publisher('scan', LaserScan, queue_size=1)
        if virtual_ugv_mode:
            localization_image_path = rospy.get_param('~localization_image_path', None)
            self.localization_image = cv2.cvtColor(cv2.imread(localization_image_path), cv2.COLOR_BGR2GRAY)
            rospy.Subscriber('/ugv_pose', Pose2D, self.virtual_pose_callback)
        else:
            rospy.Subscriber('/uav/camera/image_raw', Image, self.image_callback, queue_size=1)
            self.prev_vehicle_x = None
            self.prev_vehicle_y = None
            self.cv_bridge = CvBridge()
        if PROFILE_SCAN_GENERATOR:
            self.mean_scan_time = None
            self.scan_idx = 0
        if TRACK_NAN_SCANS:
            self.all_nans = 0
            self.any_nan = 0
        rospy.spin()


    def _publish_scan_message(self, center_x, center_y, contours_image):
        if self.prev_scan_time is None:
            self.prev_scan_time = datetime.datetime.now()
            return
        if PROFILE_SCAN_GENERATOR:
            ts = time.time()
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
        curr_scan_time = datetime.datetime.now()
        if PROFILE_SCAN_GENERATOR:
            te = time.time()
            delta = (te - ts)
            if self.scan_idx == 0:
                self.mean_scan_time = delta
            else:
                self.mean_scan_time = float(self.mean_scan_time) * (self.scan_idx - 1) / self.scan_idx + delta / self.scan_idx
            self.scan_idx += 1
            rospy.loginfo('Synthetic scan generation time: %f[sec], mean: %f[sec]' % (delta, self.mean_scan_time))
        if TRACK_NAN_SCANS:
            if np.isnan(scan_ranges).any():
                self.any_nan += 1
            if np.isnan(scan_ranges).all():
                self.all_nans += 1
                return # TODO: ???????????????????
            rospy.loginfo('Any NaN occurrences: %d' % self.any_nan)
            rospy.loginfo('All NaN occurrences: %d' % self.all_nans)
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


    def image_callback(self, message):
        # if message.header.seq % 3 == 0: # TODO: remove??
        #     return
        contours_image = self.cv_bridge.imgmsg_to_cv2(message)
        if self.prev_vehicle_x is None and self.prev_vehicle_y is None:
            vehicle_x, vehicle_y = segmentation.extract_vehicle(contours_image)
        else:
            vehicle_x, vehicle_y = segmentation.extract_vehicle(contours_image, self.prev_vehicle_x, self.prev_vehicle_y, self.max_distance * 2 + 10, self.max_distance * 2 + 10)
            # TODO: decrease the ROI size gradually!!
        self.prev_vehicle_x = vehicle_x
        self.prev_vehicle_y = vehicle_y
        contours_image = maps_generation.generate_canopies_map(contours_image, vehicle_x, vehicle_y, self.max_distance * 2 + 10, self.max_distance * 2 + 10)
        self._publish_scan_message(vehicle_x, vehicle_y, contours_image)


    def virtual_pose_callback(self, message):
        self._publish_scan_message(message.x, message.y, self.localization_image)


if __name__ == '__main__':
    SyntheticScanGenerator()