#!/usr/bin/env python

import numpy as np
import rospy
import tf.transformations
from geometry_msgs.msg import Pose2D

class SyntheticOdometry(object):
    def __init__(self):
        rospy.init_node('synthetic_odometry')
        self.broadcaster = tf.TransformBroadcaster()
        self.resolution = rospy.get_param('~resolution')
        self.prev_actual_pose = None
        self.odom_frame_id = rospy.get_param('~odom_frame_id')
        self.base_frame_id = rospy.get_param('~base_frame_id')
        self.noise_mu_x = float(rospy.get_param('~noise_mu_x', default=0)) # TODO: in meters!
        self.noise_mu_y = float(rospy.get_param('~noise_mu_y', default=0)) # TODO: in meters!
        self.noise_sigma_x = float(rospy.get_param('~noise_sigma_x', default=0)) # TODO: in meters!
        self.noise_sigma_y = float(rospy.get_param('~noise_sigma_y', default=0)) # TODO: in meters!
        rospy.Subscriber('/ugv_pose', Pose2D, self.pose_callback)

    def pose_callback(self, this_actual_pose):
        if self.prev_actual_pose is None:
            self.prev_actual_pose = this_actual_pose
            self.broadcast_values = (0, 0)
            self.broadcaster.sendTransform((0, 0, 0), tf.transformations.quaternion_from_euler(0, 0, 0), rospy.Time.now(),
                                           child=self.base_frame_id, parent=self.odom_frame_id)
            return
        actual_delta_x = this_actual_pose.x - self.prev_actual_pose.x
        actual_delta_y = this_actual_pose.y - self.prev_actual_pose.y
        if self.noise_mu_x != 0 or self.noise_sigma_x != 0:
            broadcast_delta_x = actual_delta_x * self.resolution + np.random.normal(self.noise_mu_x, self.noise_sigma_x)
        else:
            broadcast_delta_x = actual_delta_x * self.resolution
        if self.noise_mu_y != 0 or self.noise_sigma_y != 0:
            broadcast_delta_y = (actual_delta_y * self.resolution + np.random.normal(self.noise_mu_y, self.noise_sigma_y)) * (-1)
        else:
            broadcast_delta_y = (actual_delta_y * self.resolution) * (-1)
        self.broadcast_values = (self.broadcast_values[0] + broadcast_delta_x, self.broadcast_values[1] + broadcast_delta_y)
        self.broadcaster.sendTransform((self.broadcast_values[0], self.broadcast_values[1], 0), tf.transformations.quaternion_from_euler(0, 0, 0),
                                       rospy.Time.now(), child=self.base_frame_id, parent=self.odom_frame_id)
        self.prev_actual_pose = this_actual_pose


if __name__ == '__main__':
    SyntheticOdometry()
    rospy.spin()