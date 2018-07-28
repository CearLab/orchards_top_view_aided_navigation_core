#cython: boundscheck=False, wraparound=False, nonecheck=False

import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
cpdef double[:] generate(unsigned char [:, :] map_image, int center_x, int center_y, float min_angle, float max_angle, int samples_num, int min_distance, int max_distance, float resolution):

    cdef int scan_index = 0
    cdef np.ndarray scan_ranges = np.full(samples_num, np.nan, dtype=np.float)
    cdef int x, y, r
    cdef float theta
    cdef int width = np.size(map_image, 1)
    cdef int height = np.size(map_image, 0)
    cdef np.ndarray diff_vector = np.zeros(2)

    for theta in np.linspace(min_angle, max_angle, num=samples_num):
        for r in np.linspace(min_distance, max_distance):
            x = int(np.round(center_x + r * np.cos(-theta)))
            y = int(np.round(center_y + r * np.sin(-theta)))
            if not (0 <= x < width and 0 <= y < height):
                break
            if map_image[y, x] != 0: # TODO: verify order
                # scan_ranges[scan_index] = (np.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)) * resolution
                diff_vector[0] = x - center_x
                diff_vector[1] = y - center_y
                scan_ranges[scan_index] = np.linalg.norm(diff_vector) * resolution
                # scan_ranges[scan_index] = np.linalg.norm((x - center_x, y - center_y)) * resolution
                break
        scan_index += 1
    return scan_ranges