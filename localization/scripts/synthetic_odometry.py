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
        self.odom_frame_id = 'odom'
        self.base_frame_id = 'base_link'
        self.noise_sigma = float(rospy.get_param('~noise_sigma', default=0)) # TODO: in meters!
        rospy.Subscriber('/ugv_pose', Pose2D, self.pose_callback)

    def pose_callback(self, this_actual_pose):
        if self.prev_actual_pose is None:
            self.prev_actual_pose = this_actual_pose
            self.prev_broadcast = (0, 0)
            self.broadcaster.sendTransform((0, 0, 0), tf.transformations.quaternion_from_euler(0, 0, 0), rospy.Time.now(),
                                           child=self.base_frame_id, parent=self.odom_frame_id)
            return
        actual_delta_x = this_actual_pose.x - self.prev_actual_pose.x
        actual_delta_y = this_actual_pose.y - self.prev_actual_pose.y
        if self.noise_sigma != 0:
            broadcast_delta_x = actual_delta_x * self.resolution + np.random.normal(0, self.noise_sigma)
            broadcast_delta_y = actual_delta_y * self.resolution + np.random.normal(0, self.noise_sigma)
        else:
            broadcast_delta_x = actual_delta_x * self.resolution
            broadcast_delta_y = actual_delta_y * self.resolution
        broadcast_x = self.prev_broadcast[0] + broadcast_delta_x
        broadcast_y = (self.prev_broadcast[1] + broadcast_delta_y) * (-1) # TODO: verify (-1)
        self.broadcaster.sendTransform((broadcast_x, broadcast_y, 0), tf.transformations.quaternion_from_euler(0, 0, 0), rospy.Time.now(),
                                       child=self.base_frame_id, parent=self.odom_frame_id)
        self.prev_actual_pose = this_actual_pose
        self.prev_broadcast = (broadcast_x, broadcast_y)


if __name__ == '__main__':
    SyntheticOdometry()
    rospy.spin()