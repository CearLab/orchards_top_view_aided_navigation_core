import cv2
import numpy as np

def extract(image, lower_color=None, upper_color=None):

    if lower_color is None and upper_color is None:
        green_lower_hue_degrees = 65 # was: 65
        green_lower_saturation_percent = 5
        green_lower_value_percent = 0
        green_upper_hue_degrees = 160 # was: 165
        green_upper_saturation_percent = 100
        green_upper_value_percent = 100
        lower_color = np.array([green_lower_hue_degrees / 2, green_lower_saturation_percent * 255 / 100, green_lower_value_percent * 255 / 100])
        upper_color = np.array([green_upper_hue_degrees / 2, green_upper_saturation_percent * 255 / 100, green_upper_value_percent * 255 / 100])

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv_image, lower_color, upper_color)

    _, contours, hierarchy = cv2.findContours(hsv_mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # TODO: why mask copy?
    contours = filter(lambda contour: cv2.contourArea(contour) > 100, contours) # TODO: 100

    contours_mask = cv2.drawContours(image=np.zeros(image.shape, np.uint8), contours=contours, contourIdx=-1, color=255, thickness=3)
    return contours_mask

    # image_copy = image.copy()
    # cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 3)
    # viz_utils.show_image('image', image_copy)
    # TODO: erode-dialate + remove contours whose area is small