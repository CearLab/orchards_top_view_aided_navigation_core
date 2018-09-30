import cv2
import numpy as np

from framework import cv_utils
from computer_vision import segmentation
from computer_vision import trunks_detection


def generate_cost_map(image, trunk_points_list, canopy_sigma, gaussian_scale_factor,
                      gaussian_square_size_to_sigma_ratio, gaussian_circle_radius_to_sigma_ratio, trunk_radius,
                      contours_margin_width=25, step_ratio=0.5, margin_color=200):
    gaussians = trunks_detection.get_gaussians_grid_image(trunk_points_list, canopy_sigma, image.shape[1], image.shape[0],
                                                          gaussian_scale_factor, gaussian_square_size_to_sigma_ratio, gaussian_circle_radius_to_sigma_ratio)
    contours, contours_mask = segmentation.extract_canopy_contours(image, margin_width=contours_margin_width, margin_color=margin_color)
    contours_mask = contours_mask / 255.0
    contours_mask[contours_mask == 0] = step_ratio
    cost_map = np.multiply(contours_mask, gaussians)
    cost_map = cv_utils.draw_points_on_image(cost_map, trunk_points_list, color=1, radius=int(np.round(trunk_radius)))
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
