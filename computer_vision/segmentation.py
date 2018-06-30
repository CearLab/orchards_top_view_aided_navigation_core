import cv2
import numpy as np

def extract_canopy_contours(image, lower_color=None, upper_color=None, min_area=None):  # TODO: generalize min_area (should be perhaps top N contours?)
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
    contours_mask = cv2.cvtColor(cv2.drawContours(np.zeros(image.shape, np.uint8), contours, contourIdx=-1, color=(255,255,255), thickness=-1),
                                 cv2.COLOR_RGB2GRAY)
    return contours, contours_mask

    # TODO: consider: erode-dialate


def extract_four_blue_markers(image, lower_color=None, upper_color=None):
    if lower_color is None and upper_color is None:
        azure_lower_hue_degrees = 180 # 185
        azure_lower_saturation_percent = 75 # 75
        azure_lower_value_percent = 75 # 75
        azure_upper_hue_degrees = 215 # 210
        azure_upper_saturation_percent = 100 # 100
        azure_upper_value_percent = 100 # 100
        lower_color = np.array([azure_lower_hue_degrees / 2.0, azure_lower_saturation_percent * 255.0 / 100, azure_lower_value_percent * 255.0 / 100])
        upper_color = np.array([azure_upper_hue_degrees / 2.0, azure_upper_saturation_percent * 255.0 / 100, azure_upper_value_percent * 255.0 / 100])

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv_image, lower_color, upper_color)
    height, width = hsv_mask.shape
    hsv_mask[:,:width/10.0] = 0
    hsv_mask[:,9.0*width/10:] = 0
    hsv_mask[:height/10.0,:] = 0
    hsv_mask[9.0*height/10:,:] = 0
    _, contours, hierarchy = cv2.findContours(hsv_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)[0:4]
    contours_mask = cv2.drawContours(np.zeros(image.shape, np.uint8), contours, contourIdx=-1, color=255, thickness=-1)
    return contours, contours_mask


def extract_canopies_map(image, lower_color=None, upper_color=None, min_area=None):
    contours_map = np.full((np.size(image, 0), np.size(image, 1)), 0, dtype=np.uint8)
    contours, _ = extract_canopy_contours(image, lower_color, upper_color, min_area)
    cv2.drawContours(contours_map, contours, contourIdx=-1, color=128, thickness=-1)
    cv2.drawContours(contours_map, contours, contourIdx=-1, color=255, thickness=3)
    return contours_map

