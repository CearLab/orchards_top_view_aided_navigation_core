import numpy as np

def calculate_average_trunk_radius_in_pixels(row_width_in_pixels, intra_row_distance_in_pixels, measured_row_widths_in_meters,
                                             measured_intra_row_distances_in_meters, measured_trunk_perimeters_in_meters, trunk_dilation_ratio):
    pixel_to_meter_x_ratio = row_width_in_pixels / np.mean(measured_row_widths_in_meters) # TODO: verify
    pixel_to_meter_y_ratio = intra_row_distance_in_pixels / np.mean(measured_intra_row_distances_in_meters) # TODO: verify
    mean_pixel_to_meter_ratio = np.mean([pixel_to_meter_x_ratio, pixel_to_meter_y_ratio])
    mean_trunk_perimeter_in_meters = np.mean(measured_trunk_perimeters_in_meters)
    mean_trunk_radius_in_meters = mean_trunk_perimeter_in_meters / (2 * np.pi)
    trunk_radius = int(np.round(mean_trunk_radius_in_meters * mean_pixel_to_meter_ratio * trunk_dilation_ratio))
    return trunk_radius