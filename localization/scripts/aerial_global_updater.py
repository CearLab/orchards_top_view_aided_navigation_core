#!/usr/bin/env python

from collections import OrderedDict
import numpy as np
import json
import time

import rospy
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion


MIN_IMU_MESSAGES = 300
MIN_SYNC_MESSAGES = 5
MIN_DELTA_T_BETWEEN_SYNCS = 10


class AerialGlobalUpdater(object):
    def __init__(self):
        rospy.init_node('aerial_global_updater')
        self.sync_signal_count = 0
        self.imu_messages = []
        self.update_idx = 0
        self.odom_message = None
        self.last_sync_time = -np.inf
        self.relevant_update_index = rospy.get_param('~relevant_update_index')
        ugv_poses_path = rospy.get_param('~ugv_poses_path')
        self.resolution = rospy.get_param('~resolution')
        with open(ugv_poses_path) as f:
            self.ugv_poses = json.load(f, object_pairs_hook=OrderedDict)
        self.init_pose = self.ugv_poses.values()[0]
        rospy.Subscriber('/bluetooth_teleop/joy', Joy, self.joy_callback)
        rospy.Subscriber('/microstrain/imu/data', Imu, self.imu_callback)
        rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)
        self.pose_pub = rospy.Publisher('/set_pose', PoseWithCovarianceStamped, queue_size=1)
        rospy.spin()

    def joy_callback(self, message):
        if message.buttons[8] == 1 and message.buttons[10] == 1:
            self.sync_signal_count += 1
        else:
            self.sync_signal_count = 0
        if self.sync_signal_count > MIN_SYNC_MESSAGES and time.time() - self.last_sync_time > MIN_DELTA_T_BETWEEN_SYNCS:
            self.last_sync_time = time.time()
            if self.update_idx == 0:
                self.update_idx += 1
                rospy.loginfo('Received first sync (origin)')
                return
            if self.relevant_update_index != self.update_idx:
                rospy.loginfo('Identified joystick press related to irrelevant update index #%d ==> ignoring' % self.update_idx)
                self.update_idx += 1
                return
            pose_key = self.ugv_poses.keys()[self.update_idx]
            pose = self.ugv_poses[pose_key]
            dx = (pose[0] - self.init_pose[0]) * self.resolution # TODO: order???
            dy = (pose[1] - self.init_pose[1]) * self.resolution
            alpha = (np.pi/2 - self.init_yaw) * (-1)
            x = dx * np.cos(alpha) - dy * np.sin(alpha) # TODO: scale, check alpha, check x and y on image (might need to switch x and y...)
            y = dx * np.sin(alpha) + dy * np.cos(alpha)
            odom_message = self.odom_message
            pose_message = PoseWithCovarianceStamped()
            pose_message.header.stamp = odom_message.header.stamp
            pose_message.header.frame_id = odom_message.header.frame_id
            pose_message.pose = odom_message.pose
            pose_message.pose.pose.position.x = x
            pose_message.pose.pose.position.y = y
            rospy.loginfo('Performing global update from image %s to pose (%f, %f)' % (pose_key, x, y))
            self.update_idx += 1
            self.pose_pub.publish(pose_message)

    def imu_callback(self, message):
        if len(self.imu_messages) < MIN_IMU_MESSAGES:
            self.imu_messages.append(message)
            if len(self.imu_messages) == MIN_IMU_MESSAGES:
                self.init_yaw = np.mean([euler_from_quaternion([imu_message.orientation.x, imu_message.orientation.y,
                                                                imu_message.orientation.z, imu_message.orientation.w])[2] for imu_message in self.imu_messages])
        else:
            return

    def odom_callback(self, message):
        self.odom_message = message


if __name__ == '__main__':
    AerialGlobalUpdater()

# TODO: see below
'''
Assumptions:
- UAV and UGV are with the same orientation in t=0 (this angle from the robot tells the drone its yaw)
- UAV doesn't change its orientation

* need to align the images every time
* need the scale (dim_x, dim_y of the grid)
'''