import cv2
import numpy as np
from scipy.signal import find_peaks

from computer_vision import segmentation
from framework import viz_utils
from framework import cv_utils


def find_orientation(image):
    _, contours_mask = segmentation.extract_canopy_contours(image)
    angles_to_scores = {}
    for angle in np.linspace(start=-90, stop=90, num=360): # TODO: fine tuning
        rotation_mat = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, scale=1.0) # TODO: verify order of coordinates
        rotated_contours_mask = cv2.warpAffine(contours_mask, rotation_mat, (contours_mask.shape[1], contours_mask.shape[0]))
        column_sums_vector = np.sum(rotated_contours_mask, axis=0)
        maxima_indices, _ = find_peaks(column_sums_vector, distance=200, width=50)
        minima_indices, _ = find_peaks(column_sums_vector * (-1), distance=200, width=50)
        maxima_values = [column_sums_vector[index] for index in maxima_indices]
        minima_values = [column_sums_vector[index] for index in minima_indices]
        mean_maxima = np.mean(maxima_values) if len(maxima_values) > 0 else 0
        mean_minima = np.mean(minima_values) if len(minima_values) > 0 else 1e30
        angles_to_scores[angle] = (mean_maxima, mean_minima)
    keys_maxima = [key for key, value in sorted(angles_to_scores.iteritems(), key=lambda (k, v): v[0], reverse=True)]
    keys_minima = [key for key, value in sorted(angles_to_scores.iteritems(), key=lambda (k, v): v[1], reverse=False)]
    scores = {}
    for angle in angles_to_scores.keys():
        scores[angle] = keys_minima.index(angle)
    orientation = [key for key, value in sorted(scores.iteritems(), key=lambda (k, v): v)][0]
    # TODO: this function is a complete mess (code and logic)
    # TODO: convert to fourier logic
    return orientation


def find_tree_centroids(image, angle=None):
    if angle is not None:
        rotation_mat = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, scale=1.0)  # TODO: verify order of coordinates
        image = cv2.warpAffine(image, rotation_mat, (image.shape[1], image.shape[0]))
    centroids = []
    _, contours_mask = segmentation.extract_canopy_contours(image)
    column_sums_vector = np.sum(contours_mask, axis=0)
    aisle_centers, _ = find_peaks(column_sums_vector * (-1), distance=200, width=50)
    for tree_row_left_limit, tree_tow_right_limit in zip(aisle_centers[:-1], aisle_centers[1:]):
        tree_row = contours_mask[:, tree_row_left_limit:tree_tow_right_limit]
        row_sums_vector = np.sum(tree_row, axis=1)
        tree_locations_in_row, _ = find_peaks(row_sums_vector, distance=160, width=30)
        centroids.append([(int(np.mean([tree_row_left_limit, tree_tow_right_limit])), tree_location) for tree_location in tree_locations_in_row])
    return centroids


def estimate_deltas(rotated_centroids, angle=None):
    if angle is not None:
        raise NotImplementedError
    # TODO: rotate if angle is given
    row_locations = map(lambda row: np.median([centroid[0] for centroid in row]), rotated_centroids)
    delta_x = np.median(np.array(row_locations[1:]) - np.array(row_locations[:-1]))
    tree_distances = []
    for row in rotated_centroids:
        tree_locations = sorted([centroid[1] for centroid in row])
        tree_distances += list(np.array(tree_locations[1:]) - np.array(tree_locations[:-1]))
    delta_y = np.median(tree_distances)
    return delta_x, delta_y

def estimate_shear(rotated_centoids, angle=None):
    if angle is not None:
        raise NotImplementedError
    # TODO: rotate if angle is given
    lines = []
    thetas = []
    for row, next_row in zip(rotated_centoids[:-1], rotated_centoids[1:]):
        print 'new row'
        for centroid in row:
            distances = [(centroid[0] - next_row_centroid[0]) ** 2 + (centroid[1] - next_row_centroid[1]) ** 2 for next_row_centroid in next_row]
            closest_centroid = next_row[distances.index(min(distances))]
            # print np.rad2deg((np.arctan2(centroid[1], centroid[0]) - np.arctan2(closest_centroid[1], closest_centroid[0])) % (2 * np.pi))
            # print np.rad2deg(np.arctan2(closest_centroid[1] - centroid[1], closest_centroid[0] - centroid[0]))
            thetas.append(np.rad2deg(np.arctan2(closest_centroid[1] - centroid[1], closest_centroid[0] - centroid[0]) * (-1))) # TODO: verify correctness (especially CCW)
            lines.append((centroid, closest_centroid))
    shear = np.median(thetas)
    # TODO: can use RANSAC / linear regression / PCA to estimate the shear
    return shear