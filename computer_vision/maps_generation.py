import cv2
import numpy as np

from framework import cv_utils
from computer_vision import segmentation
from computer_vision import trunks_detection

def generate_cost_map(image, trunk_points_list, canopy_sigma, gaussian_scale_factor, gaussian_square_size_to_sigma_ratio, gaussian_circle_radius_to_sigma_ratio, trunk_radius):
    trunk_points_list = [(int(elem[0]), int(elem[1])) for elem in trunk_points_list]
    upper_left, lower_right = cv_utils.get_bounding_box(image, trunk_points_list, expand_ratio=0.1)
    cropped_image = image[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
    trunk_points_list = np.array(trunk_points_list) - np.array(upper_left)
    gaussians = trunks_detection.get_gaussians_grid_image(trunk_points_list, canopy_sigma, cropped_image.shape[1], cropped_image.shape[0],
                                                          gaussian_scale_factor, gaussian_square_size_to_sigma_ratio, gaussian_circle_radius_to_sigma_ratio)
    contours, contours_mask = segmentation.extract_canopy_contours(cropped_image)
    cost_map = cv2.bitwise_and(gaussians, gaussians, mask=contours_mask)
    cost_map = cv_utils.draw_points_on_image(cost_map, trunk_points_list, color=1, radius=trunk_radius)
    return cost_map, upper_left
