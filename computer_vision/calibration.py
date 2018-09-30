import numpy as np

def calculate_trunk_radius_in_meters(measured_trunk_perimeters_in_meters):
    approximated_trunk_radii_in_meters = map(lambda perimeter: perimeter / (2 * np.pi), measured_trunk_perimeters_in_meters)
    mean_trunk_radius_in_meters = np.mean(approximated_trunk_radii_in_meters)
    std_trunk_radius_in_meters = np.std(approximated_trunk_radii_in_meters)
    return mean_trunk_radius_in_meters, std_trunk_radius_in_meters


def calculate_pixel_to_meter_ratio(grid_dim_x_in_pixels, grid_dim_y_in_pixels, measured_row_widths_in_meters, measured_intra_row_distances_in_meters):
    mean_row_width_in_meters = np.mean(measured_row_widths_in_meters)
    mean_intra_row_distance_in_meters = np.mean(measured_intra_row_distances_in_meters)
    return float(np.mean([float(grid_dim_x_in_pixels) / float(mean_row_width_in_meters), float(grid_dim_y_in_pixels) / float(mean_intra_row_distance_in_meters)]))
    # TODO: for this calculation to be valid, you need to show that camera's Fx ~ Fy (calibration output)