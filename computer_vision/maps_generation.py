import cv2
import numpy as np

from computer_vision import segmentation


def generate_cost_map(image):
    contours, contours_mask = segmentation.extract_canopy_contours(image)
    cost_map = np.full((np.size(image, 0), np.size(image, 1)), fill_value=0, dtype=np.uint8)
    cv2.drawContours(cost_map, contours, contourIdx=-1, color=255, thickness=-1)
    cv2.drawContours(cost_map, contours, contourIdx=-1, color=150, thickness=90)
    cost_map = np.minimum(contours_mask, cost_map)
    cv2.drawContours(cost_map, contours, contourIdx=-1, color=50, thickness=28)
    cost_map = cost_map / 255.0
    return cost_map


def generate_canopies_map(image, lower_color=None, upper_color=None, min_area=None):
    contours_map = np.full((np.size(image, 0), np.size(image, 1)), fill_value=0, dtype=np.uint8)
    contours, _ = segmentation.extract_canopy_contours(image, lower_color, upper_color, min_area)
    cv2.drawContours(contours_map, contours, contourIdx=-1, color=128, thickness=-1)
    cv2.drawContours(contours_map, contours, contourIdx=-1, color=255, thickness=3)
    return contours_map


def generate_trunks_map(image, trunk_points_list, mean_trunk_radius, std_trunk_radius, np_random_state=None):
    if np_random_state is not None:
        np.random.set_state(np_random_state)
    trunks_map = np.full((np.size(image, 0), np.size(image, 1)), fill_value=0, dtype=np.uint8)
    for trunk_center in trunk_points_list:
        center_x = int(np.round(trunk_center[0]))
        center_y = int(np.round(trunk_center[1]))
        if int(np.round(std_trunk_radius)) == 0:
            first_axis = 0
            second_axis = 0
        else:
            first_axis = int(np.round(mean_trunk_radius)) + np.random.randint(-int(np.round(std_trunk_radius)), int(np.round(std_trunk_radius)))
            second_axis = int(np.round(mean_trunk_radius)) + np.random.randint(-int(np.round(std_trunk_radius)), int(np.round(std_trunk_radius)))
        angle = np.random.randint(0, 360)
        cv2.ellipse(trunks_map, center=(center_x, center_y), axes=(first_axis, second_axis), angle=angle, startAngle=0, endAngle=360, color=128, thickness=-1)
        cv2.ellipse(trunks_map, center=(center_x, center_y), axes=(first_axis, second_axis), angle=angle, startAngle=0, endAngle=360, color=255, thickness=3)
    return trunks_map
