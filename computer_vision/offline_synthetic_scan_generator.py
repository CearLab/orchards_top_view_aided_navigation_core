import numpy as np
import pickle
import rosbag

from computer_vision.contours_scan_cython import contours_scan

def generate_scans_pickle(trajectory_bag_path, localization_image, min_angle, max_angle, samples_num, min_distance, max_distance,
                          resolution, r_primary_search_samples, r_secondary_search_step, output_pickle_path):
    trajectory_bag = rosbag.Bag(trajectory_bag_path)
    timestamp_to_scan = {}
    for _, message, timestamp in trajectory_bag.read_messages(topics='ugv_pose'):
        scan_ranges = contours_scan.generate(localization_image,
                                             center_x=message.point.x,
                                             center_y=message.point.y,
                                             min_angle=min_angle,
                                             max_angle=max_angle,
                                             samples_num=samples_num,
                                             min_distance=min_distance,
                                             max_distance=max_distance,
                                             resolution=resolution,
                                             r_primary_search_samples=r_primary_search_samples,
                                             r_secondary_search_step=r_secondary_search_step)
        timestamp_to_scan[message.header.stamp] = np.asanyarray(scan_ranges)
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(timestamp_to_scan, f)