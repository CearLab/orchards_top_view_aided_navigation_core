#!/usr/bin/env python

import cv2
import pandas as pd

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped
import tf.transformations


class ScanPoseOdomPlayback(object):

    def __init__(self):
        rospy.init_node('scan_pose_odom_playback')
        scans_and_poses_pickle_path = rospy.get_param('~scans_and_poses_pickle_path')
        odom_pickle_path = rospy.get_param('~odom_pickle_path')
        scans_poses_df = pd.DataFrame.from_dict(pd.read_pickle(scans_and_poses_pickle_path), orient='index', columns=['scan', 'pose'])
        odom_df = pd.DataFrame.from_dict(pd.read_pickle(odom_pickle_path), orient='index', columns=['odom_x', 'odom_y'])
        self.joint_df = pd.concat([scans_poses_df, odom_df], axis=1).fillna(method='ffill')
        self.video_path = rospy.get_param('~video_path')
        self.scan_publisher = rospy.Publisher('scan', LaserScan, queue_size=1)
        self.pose_publisher = rospy.Publisher('ugv_pose', PointStamped, queue_size=1)
        self.odom_broadcaster = tf.TransformBroadcaster()

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_idx = 0
        frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        while cap.isOpened():
            is_success, frame = cap.read()
            frame_idx += 1
            if frame_idx >= frames_count:
                break
            video_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) * 1e-3
            if not is_success:
                continue
            this_index = self.joint_df.index.get_loc(video_timestamp, method='nearest')
            this_timestamp_data = self.joint_df.iloc[this_index]
            odom_x = this_timestamp_data['odom_x']
            odom_y = this_timestamp_data['odom_y']
            scan = this_timestamp_data['scan']
            pose = this_timestamp_data['pose']
            if pd.isnull(scan) or pd.isnull(pose) or pd.isnull(odom_x) or pd.isnull(odom_y):
                continue
            scan.header.stamp = rospy.rostime.Time.now()
            pose.header.stamp = rospy.rostime.Time.now()

            self.scan_publisher.publish(scan)
            self.pose_publisher.publish(pose)
            self.odom_broadcaster.sendTransform((odom_x, odom_y, 0), tf.transformations.quaternion_from_euler(0, 0, 0),
                                                rospy.Time.now(), child='canopies/base_link', parent='canopies/odom')
            rospy.sleep(1.0/40) # TODO: ???
        cap.release()


if __name__ == '__main__':
    playback = ScanPoseOdomPlayback()
    playback.run()