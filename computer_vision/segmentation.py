import cv2
import numpy as np

from framework import cv_utils

def extract_canopy_contours(image, lower_color=None, upper_color=None, min_area=None, margin_width=0, margin_color=None):  # TODO: generalize min_area (should be perhaps top N contours?)
    if lower_color is None and upper_color is None:
        green_lower_hue_degrees = 65
        green_lower_saturation_percent = 5
        green_lower_value_percent = 0
        green_upper_hue_degrees = 160
        green_upper_saturation_percent = 100
        green_upper_value_percent = 100
        lower_color = np.array([green_lower_hue_degrees / 2.0, green_lower_saturation_percent * 255.0 / 100, green_lower_value_percent * 255.0 / 100])
        upper_color = np.array([green_upper_hue_degrees / 2.0, green_upper_saturation_percent * 255.0 / 100, green_upper_value_percent * 255.0 / 100])
    if min_area is None:
        min_area = 10000
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv_image, lower_color, upper_color)
    _, contours, hierarchy = cv2.findContours(hsv_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = filter(lambda contour: cv2.contourArea(contour) > min_area, contours)
    contours_mask = cv2.drawContours(np.zeros((image.shape[0], image.shape[1]), np.uint8), contours, contourIdx=-1, color=255, thickness=-1)
    if margin_width != 0: # TODO: consider moving this part to the calling functions (map generation)
        contours_mask = cv2.drawContours(np.zeros((image.shape[0], image.shape[1]), np.uint8), contours, contourIdx=-1, color=margin_color, thickness=margin_width * 2)
        contours_mask = cv2.drawContours(contours_mask, contours, contourIdx=-1, color=255, thickness=-1)
    return contours, contours_mask


def extract_vehicle(image, roi_center_x=None, roi_center_y=None, roi_size_x=None, roi_size_y=None,
                    lower_color=None, upper_color=None):
    if lower_color is None and upper_color is None:
        purple_lower_hue_degrees = 210
        purple_lower_saturation_percent = 60
        purple_lower_value_percent = 60
        purple_upper_hue_degrees = 230
        purple_upper_saturation_percent = 100
        purple_upper_value_percent = 100
        lower_color = np.array([purple_lower_hue_degrees / 2.0, purple_lower_saturation_percent * 255.0 / 100, purple_lower_value_percent * 255.0 / 100])
        upper_color = np.array([purple_upper_hue_degrees / 2.0, purple_upper_saturation_percent * 255.0 / 100, purple_upper_value_percent * 255.0 / 100])
    if roi_center_x is not None and roi_center_y is not None:
        image, roi_upper_left, roi_lower_right = cv_utils.crop_region(image, roi_center_x, roi_center_y, roi_size_x, roi_size_y)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv_image, lower_color, upper_color)
    hsv_mask = cv2.dilate(hsv_mask, kernel=np.ones((9,9),np.uint8), iterations=1)
    _, contours, hierarchy = cv2.findContours(hsv_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    largest_contour = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)[0]
    moments = cv2.moments(largest_contour)
    if roi_center_x is not None and roi_center_y is not None:
        contour_center_x = roi_upper_left[0] + int(moments['m10'] / moments['m00'])
        contour_center_y = roi_upper_left[1] + int(moments['m01'] / moments['m00'])
    else:
        contour_center_x = int(moments['m10'] / moments['m00'])
        contour_center_y = int(moments['m01'] / moments['m00'])
    return contour_center_x, contour_center_y
