#!/usr/bin/env python

import cv2
import numpy as np
import pickle
import rospy
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from framework import cv_utils

alpha = 0.3
downsample_rate = 3

class ContoursScanVisualizer(object):
    def __init__(self):
        rospy.init_node('contours_scan_visualizer')
        canopies_image_path = rospy.get_param('~canopies_image_path')
        trunks_image_path = rospy.get_param('~trunks_image_path')
        canopies_scans_pickle_path = rospy.get_param('~canopies_scans_pickle_path')
        trunks_scans_pickle_path = rospy.get_param('~trunks_scans_pickle_path')
        canopies_image = cv2.imread(canopies_image_path)
        trunks_image = cv2.imread(trunks_image_path)
        self.min_angle = rospy.get_param('~min_angle')
        self.max_angle = rospy.get_param('~max_angle')
        self.resolution = rospy.get_param('~resolution')
        self.window_size = rospy.get_param('~window_size')
        self.base_viz_image = self._prepare_base_viz_image(canopies_image, trunks_image)
        with open(canopies_scans_pickle_path) as f:
            self.canopies_timestamp_to_scan = pickle.load(f)
        with open(trunks_scans_pickle_path) as f:
            self.trunks_timestamp_to_scan = pickle.load(f)
        self.pose_idx = 0
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher('scan_visualization', Image, queue_size=1)
        rospy.Subscriber('/ugv_pose', PointStamped, self.pose_callback)

    def _prepare_base_viz_image(self, canopies_image, trunks_image):
        trunks_gray = cv2.cvtColor(trunks_image, cv2.COLOR_BGR2GRAY)
        trunks_gray[trunks_gray < 30] = 0
        trunks_gray[trunks_gray > 220] = 255
        trunks_brown = trunks_image.copy()
        trunks_brown[trunks_gray != 0] = (0, 47, 95)
        trunks_brown[trunks_gray == 255] = (0, 47, 130)
        weighted = cv2.addWeighted(canopies_image, alpha, trunks_brown, 1 - alpha, gamma=0)
        update_indices = np.where(trunks_gray != 0)
        canopies_image[update_indices] = weighted[update_indices]
        return canopies_image

    def pose_callback(self, message):
        self.pose_idx = (self.pose_idx + 1) % downsample_rate
        if self.pose_idx != 0:
            return
        canopies_scan_ranges = self.canopies_timestamp_to_scan[message.header.stamp]
        canopies_scan_points_list = cv_utils.get_coordinates_list_from_scan_ranges(canopies_scan_ranges, message.point.x, message.point.y,
                                                                                   self.min_angle, self.max_angle, self.resolution)
        viz_image = cv_utils.draw_points_on_image(self.base_viz_image, canopies_scan_points_list, color=(0, 0, 255), radius=3)
        trunks_scan_ranges = self.trunks_timestamp_to_scan[message.header.stamp]
        trunks_scan_points_list = cv_utils.get_coordinates_list_from_scan_ranges(trunks_scan_ranges, message.point.x, message.point.y,
                                                                                 self.min_angle, self.max_angle, self.resolution)
        viz_image = cv_utils.draw_points_on_image(viz_image, trunks_scan_points_list, color=(255, 255, 0), radius=3)
        cv2.circle(viz_image, (int(np.round(message.point.x)), int(np.round(message.point.y))), radius=5, color=(255, 0, 255), thickness=-1)
        viz_image, _, _ = cv_utils.crop_region(viz_image, message.point.x, message.point.y, self.window_size, self.window_size) # TODO: read from config
        image_message = self.bridge.cv2_to_imgmsg(viz_image, encoding='bgr8')
        self.image_pub.publish(image_message)


if __name__ == '__main__':
    ContoursScanVisualizer()
    rospy.spin()