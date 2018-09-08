#!/usr/bin/env python

import numpy as np
import datetime
import time
import cv2
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose2D

from air_ground_orchard_navigation.computer_vision.contours_scan_cython import contours_scan

PROFILE_SCAN_GENERATOR = False

class SyntheticScanGenerator(object):
    def __init__(self):
        rospy.init_node('synthetic_scan_generator')
        image_path = rospy.get_param('~localization_image_path')
        self.localization_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_RGB2GRAY)
        self.frame_id = 'contours_scan_link'
        self.min_angle = 0
        self.max_angle = 2 * np.pi
        self.samples_num = 360
        self.min_distance = 3
        self.max_distance = 300
        self.resolution = rospy.get_param('~resolution')
        self.prev_scan_time = None
        self.scan_pub = rospy.Publisher('scan', LaserScan, queue_size=1) # TODO: queue size=?
        if PROFILE_SCAN_GENERATOR:
            self.mean_scan_time = None
            self.no_scans = 0
        rospy.Subscriber('/vehicle_pose', Pose2D, self.pose_callback)
        rospy.spin()


    def pose_callback(self, message):
        if self.prev_scan_time is None:
            self.prev_scan_time = datetime.datetime.now()
            return
        # TODO: check if pixel value in (center_x, center_y) is not black and if yes - return (don't execute further this iteration)
        ts = time.time()
        scan_ranges = contours_scan.generate(self.localization_image,
                                             center_x=message.x,
                                             center_y=message.y,
                                             min_angle=self.min_angle,
                                             max_angle=self.max_angle,
                                             samples_num=self.samples_num,
                                             min_distance=self.min_distance,
                                             max_distance=self.max_distance,
                                             resolution=self.resolution)  # TODO: fine tune parameters!
        te = time.time()
        if PROFILE_SCAN_GENERATOR:
            delta = (te - ts)
            if self.no_scans == 0:
                self.mean_scan_time = delta
            else:
                self.mean_scan_time = float(self.mean_scan_time) * (self.no_scans - 1) / self.no_scans + delta / self.no_scans
            self.no_scans += 1
            rospy.loginfo('Synthetic scan generation time: %f[sec], mean: %f[sec]' % (delta, self.mean_scan_time))
        laser_scan = LaserScan()
        laser_scan.header.stamp = rospy.rostime.Time.now()
        laser_scan.header.frame_id = self.frame_id
        laser_scan.angle_min = self.min_angle
        laser_scan.angle_max = self.max_angle
        laser_scan.angle_increment = (self.max_angle - self.min_angle) / self.samples_num
        laser_scan.scan_time = (datetime.datetime.now() - self.prev_scan_time).seconds
        laser_scan.range_min = self.min_distance * self.resolution
        laser_scan.range_max = self.max_distance * self.resolution
        laser_scan.ranges = scan_ranges
        self.scan_pub.publish(laser_scan)


if __name__ == '__main__':
    SyntheticScanGenerator()