#cython: boundscheck=False, wraparound=False, nonecheck=False

import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
cpdef double[:] generate(unsigned char [:, :] map_image, int center_x, int center_y, float min_angle, float max_angle,
                         int samples_num, int min_distance, int max_distance, float resolution,
                         int r_primary_search_samples, int r_secondary_search_step):


    cdef int scan_index = 0
    cdef np.ndarray scan_ranges = np.full(samples_num, np.nan, dtype=np.float)
    cdef int x, y
    cdef float theta, r_primary, r_secondary
    cdef int width = np.size(map_image, 1)
    cdef int height = np.size(map_image, 0)
    cdef np.ndarray theta_values = np.linspace(min_angle, max_angle, num=samples_num)
    cdef np.ndarray r_primary_search_values = np.linspace(min_distance, max_distance, num=r_primary_search_samples)
    cdef np.ndarray r_secondary_search_values
    cdef np.ndarray diff_vector = np.zeros(2)

    if map_image[center_y, center_x] != 0:
        return scan_ranges

    for theta in theta_values:
        for r_primary in r_primary_search_values:
            x = int(np.round(center_x + r_primary * np.cos(-theta)))
            y = int(np.round(center_y + r_primary * np.sin(-theta)))
            if not (0 <= x < width and 0 <= y < height):
                break
            if map_image[y, x] != 0:
                diff_vector[0] = x - center_x
                diff_vector[1] = y - center_y
                r_secondary_search_values = np.arange(start=-np.round((max_distance - min_distance) / r_primary_search_samples),
                                                      stop=-1, step=r_secondary_search_step) + r_primary
                for r_secondary in r_secondary_search_values:
                    x = int(np.round(center_x + r_secondary * np.cos(-theta)))
                    y = int(np.round(center_y + r_secondary * np.sin(-theta)))
                    if map_image[y, x] != 0:
                        diff_vector[0] = x - center_x
                        diff_vector[1] = y - center_y
                        break
                scan_ranges[scan_index] = np.linalg.norm(diff_vector) * resolution
                break
        scan_index += 1
    return scan_ranges