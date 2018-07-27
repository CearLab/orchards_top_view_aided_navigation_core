#!/usr/bin/env python

import numpy as np
import datetime
import cv2
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose2D

from air_ground_orchard_navigation.computer_vision import contours_scan2
from air_ground_orchard_navigation.computer_vision.contours_scan_cython import contours_scan


class SyntheticScanGenerator(object):
    def __init__(self):
        rospy.init_node('synthetic_scan_generator')
        image_path = rospy.get_param('~map_image_path')
        self.map_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_RGB2GRAY)
        self.frame_id = 'contours_scan_link'
        self.min_angle = 0
        self.max_angle = 2 * np.pi
        self.samples_num = 360
        self.min_distance = 3
        self.max_distance = 300
        self.resolution = 0.0125
        self.prev_scan_time = None
        self.idx = 0
        self.scan_pub = rospy.Publisher('scan', LaserScan, queue_size=1)
        rospy.Subscriber('/vehicle_pose', Pose2D, self.pose_callback)
        rospy.spin()


    def pose_callback(self, message):
        if self.prev_scan_time is None:
            self.prev_scan_time = datetime.datetime.now()
            return
        if self.idx == 1:
            self.idx = 0
            return
        self.idx = 1
        # TODO: check if pixel value in (center_x, center_y) is not black and if yes - return (don't execute further this iteration)
        before1 = datetime.datetime.now()
        scan_ranges, _ = contours_scan2.generate(self.map_image,
                                             center_x=message.x,
                                             center_y=message.y,
                                             min_angle=self.min_angle,
                                             max_angle=self.max_angle,
                                             samples_num=self.samples_num,
                                             min_distance=self.min_distance,
                                             max_distance=self.max_distance,
                                             resolution=self.resolution)  # TODO: fine tune parameters!
        after1 = datetime.datetime.now()
        print ('delta1 = ' + str((after1-before1).microseconds))
        before2 = datetime.datetime.now()
        scan_ranges = contours_scan.generate(self.map_image,
                                             center_x=message.x,
                                             center_y=message.y,
                                             min_angle=self.min_angle,
                                             max_angle=self.max_angle,
                                             samples_num=self.samples_num,
                                             min_distance=self.min_distance,
                                             max_distance=self.max_distance,
                                             resolution=self.resolution)  # TODO: fine tune parameters!
        after2 = datetime.datetime.now()
        print ('delta2 = ' + str((after2-before2).microseconds))
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
    # import cv2

    # key = '19-03-5'
    # data_descriptor = dji.snapshots_60_meters[key]
    # image = cv2.imread(data_descriptor.path)
    # map_image = segmentation.extract_canopies_map(image)
    # map_image = cv2.imread(r'/home/omer/Downloads/temp_map_2.pgm')
    # map_image = cv2.cvtColor(map_image, cv2.COLOR_RGB2GRAY)

    # SyntheticScanGenerator(map_image)