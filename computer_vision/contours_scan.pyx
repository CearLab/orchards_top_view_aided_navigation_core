#cython: boundscheck=False, wraparound=False, nonecheck=False

import cython
import numpy as np
from cython.parallel import prange
cimport numpy as np

@cython.boundscheck(False)
cpdef double[:] generate(unsigned char [:, :] map_image, int center_x, int center_y, float min_angle, float max_angle, int samples_num, int min_distance, int max_distance, float resolution):

    cdef int scan_index = 0
    cdef np.ndarray scan_ranges = np.full(samples_num, np.nan, dtype=np.float)
    cdef int x, y, r
    cdef float theta

    for theta in np.linspace(start=min_angle, stop=max_angle, num=samples_num):
        for r in np.arange(start=min_distance, stop=max_distance, step=1):
            x = int(np.round(center_x + r * np.cos(-theta)))
            y = int(np.round(center_y + r * np.sin(-theta)))
            if not (0 <= x < np.size(map_image, 1) and 0 <= y < np.size(map_image, 0)):
                break
            if map_image[y, x] in [255, 128]: # TODO: verify order
                # scan_ranges[scan_index] = (np.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)) * resolution
                scan_ranges[scan_index] = np.linalg.norm((x - center_x, y - center_y)) * resolution
                break
            # if map_image[y, x] != 0: # TODO: is that condition needed?
            #     break
        scan_index += 1
    return scan_ranges