import itertools
from collections import OrderedDict
import cv2
import numpy as np
from scipy.signal import find_peaks

from computer_vision import segmentation
from framework import viz_utils
from framework import cv_utils


def estimate_rows_orientation(image):
    _, contours_mask = segmentation.extract_canopy_contours(image)
    angles_to_scores = {}
    for correction_angle in np.linspace(start=-90, stop=90, num=360): # TODO: fine tuning
        rotation_mat = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), correction_angle, scale=1.0) # TODO: verify order of coordinates
        rotated_contours_mask = cv2.warpAffine(contours_mask, rotation_mat, (contours_mask.shape[1], contours_mask.shape[0]))
        column_sums_vector = np.sum(rotated_contours_mask, axis=0)
        # TODO: hardcoded numbers are bad!!!
        maxima_indices, _ = find_peaks(column_sums_vector, distance=200, width=50)
        minima_indices, _ = find_peaks(column_sums_vector * (-1), distance=200, width=50)
        maxima_values = [column_sums_vector[index] for index in maxima_indices]
        minima_values = [column_sums_vector[index] for index in minima_indices]
        mean_maxima = np.mean(maxima_values) if len(maxima_values) > 0 else 0
        mean_minima = np.mean(minima_values) if len(minima_values) > 0 else 1e30
        angles_to_scores[correction_angle] = (mean_maxima, mean_minima)
    keys_maxima = [key for key, value in sorted(angles_to_scores.iteritems(), key=lambda (k, v): v[0], reverse=True)]
    keys_minima = [key for key, value in sorted(angles_to_scores.iteritems(), key=lambda (k, v): v[1], reverse=False)]
    scores = {}
    for correction_angle in angles_to_scores.keys():
        scores[correction_angle] = keys_minima.index(correction_angle)
    orientation = [key for key, value in sorted(scores.iteritems(), key=lambda (k, v): v)][0] * (-1)
    # TODO: this function is a complete mess (code and logic)
    # TODO: convert to fourier logic
    return orientation


def find_tree_centroids(image, correction_angle):
    rotation_mat = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), correction_angle, scale=1.0)  # TODO: verify order of coordinates
    rotated_image = cv2.warpAffine(image, rotation_mat, (image.shape[1], image.shape[0]))
    rotated_centroids = []
    _, contours_mask = segmentation.extract_canopy_contours(rotated_image)
    column_sums_vector = np.sum(contours_mask, axis=0)
    aisle_centers, _ = find_peaks(column_sums_vector * (-1), distance=200, width=50)
    for tree_row_left_limit, tree_tow_right_limit in zip(aisle_centers[:-1], aisle_centers[1:]):
        tree_row = contours_mask[:, tree_row_left_limit:tree_tow_right_limit]
        row_sums_vector = np.sum(tree_row, axis=1)
        tree_locations_in_row, _ = find_peaks(row_sums_vector, distance=160, width=30)
        rotated_centroids.append([(int(np.mean([tree_row_left_limit, tree_tow_right_limit])), tree_location) for tree_location in tree_locations_in_row])
    vertical_rows_centroids_np = np.float32(list(itertools.chain.from_iterable(rotated_centroids))).reshape(-1, 1, 2)
    rotation_mat = np.insert(cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), correction_angle * (-1), scale=1.0), [2], [0, 0, 1], axis=0)  # TODO: verify coordinates order
    centroids_np = cv2.perspectiveTransform(vertical_rows_centroids_np, rotation_mat)
    centroids = [tuple(elem) for elem in centroids_np[:, 0, :].tolist()]
    return centroids, rotated_centroids


def estimate_grid_dimensions(rotated_centroids):
    row_locations = map(lambda row: np.median([centroid[0] for centroid in row]), rotated_centroids)
    dim_x = np.median(np.array(row_locations[1:]) - np.array(row_locations[:-1]))
    tree_distances = []
    for row in rotated_centroids:
        tree_locations = sorted([centroid[1] for centroid in row])
        tree_distances += list(np.array(tree_locations[1:]) - np.array(tree_locations[:-1]))
    dim_y = np.median(tree_distances)
    return dim_x, dim_y


def estimate_shear(rotated_centoids):
    drift_vectors = []
    thetas = []
    for row, next_row in zip(rotated_centoids[:-1], rotated_centoids[1:]):
        for centroid in row:
            distances = [(centroid[0] - next_row_centroid[0]) ** 2 + (centroid[1] - next_row_centroid[1]) ** 2 for next_row_centroid in next_row]
            closest_centroid = next_row[distances.index(min(distances))]
            thetas.append(np.arctan2(closest_centroid[1] - centroid[1], closest_centroid[0] - centroid[0]))
            drift_vectors.append((centroid, closest_centroid))
    def reject_outliers(data, m):
        data_np = np.array(data)
        data_np = data_np[abs(data_np - np.median(data_np)) < m * np.std(data_np)]
        return data_np.tolist()
    thetas_filtered = reject_outliers(thetas, m=1.5) # TODO: perhaps it's too agressive and I'm losing inliers!
    drift_vectors = [drift_vectors[index] for index in [thetas.index(theta) for theta in thetas_filtered]]
    shear = np.sin(np.median(thetas_filtered))
    return shear, drift_vectors


def get_essential_grid(dim_x, dim_y, shear, orientation, n):
    nodes = list(itertools.product(np.arange(0, n * dim_x, step=dim_x), np.arange(0, n * dim_y, step=dim_y)))
    nodes_np = np.float32(nodes).reshape(-1, 1, 2)
    shear_mat = np.float32([[1, 0, 0], [shear, 1, 0], [0, 0, 1]])
    transformed_nodes_np = cv2.perspectiveTransform(nodes_np, shear_mat)
    rotation_center = tuple(np.mean(transformed_nodes_np, axis=0)[0])
    rotation_mat = np.insert(cv2.getRotationMatrix2D(rotation_center, orientation, scale=1.0), [2], [0, 0, 1], axis=0) # TODO: verify coordinates order
    transformed_nodes_np = cv2.perspectiveTransform(transformed_nodes_np, rotation_mat)
    transformed_nodes = [tuple(elem) for elem in transformed_nodes_np[:, 0, :].tolist()]
    return transformed_nodes


def find_min_mse_position(centroids, grid, image_width, image_height):
    min_error = np.inf
    optimal_translation = None
    optimal_drift_vectors = None
    optimal_grid = None
    for candidate_translation in centroids:
        candidate_grid_np = np.array(grid) - grid[0] + candidate_translation
        if np.any(candidate_grid_np < 0) or np.any(candidate_grid_np[:,0] >= image_height) or np.any(candidate_grid_np[:,1] >= image_width): # TODO: verify this! potential logic bug which won't fail!!!!!!!!!!!
            continue
        error = 0
        drift_vectors = []
        for x, y in candidate_grid_np:
            distances = [(x - centroid[0]) ** 2 + (y - centroid[1]) ** 2 for centroid in centroids]
            error += min(distances)
            drift_vectors.append(((x, y), centroids[distances.index(min(distances))]))
        if error < min_error:
            min_error = error
            optimal_translation = tuple(np.array(candidate_translation) - np.array(grid[0]))
            optimal_drift_vectors = drift_vectors
            optimal_grid = [tuple(elem) for elem in candidate_grid_np.tolist()]
    return optimal_grid, optimal_translation, optimal_drift_vectors


def get_grid(dim_x, dim_y, translation, orientation, shear, n):
    essential_grid = get_essential_grid(dim_x, dim_y, shear, orientation, n)
    essential_grid_np = np.float32(essential_grid).reshape(-1, 1, 2)
    translation_mat = np.float32([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]]) # TODO: verify correctenss!
    transformed_grid_np = cv2.perspectiveTransform(essential_grid_np, translation_mat)
    transformed_grid = [tuple(elem) for elem in transformed_grid_np[:, 0, :].tolist()]
    return transformed_grid


def get_gaussian_on_image(mu_x, mu_y, sigma, image_width, image_height):
    gaussian_image = np.full((image_height, image_width), fill_value=0, dtype=np.float64)
    square_size = 2 * sigma
    circle_radius = 1.5 * sigma
    x_start, x_end = max(0, int(mu_x - square_size)), min(image_width, int(mu_x + square_size)) # TODO: width
    y_start, y_end = max(0, int(mu_y - square_size)), min(image_height, int(mu_y + square_size)) # TODO: height
    x, y = np.meshgrid(np.arange(x_start, x_end), np.arange(y_start, y_end))
    squre_gaussian = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2.0 * sigma ** 2))
    circle_mask = cv2.circle(img=np.full(squre_gaussian.shape, fill_value=0.0, dtype=np.float64),
                             center=(int(mu_x - x_start), int(mu_y - y_start)), radius=int(circle_radius), color=1.0, thickness=-1)
    circle_gaussian = np.multiply(squre_gaussian, circle_mask)
    gaussian_image = cv_utils.insert_image_patch(gaussian_image, circle_gaussian, upper_left=(x_start, y_start), lower_right=(x_end, y_end))
    return gaussian_image


def get_gaussians_grid_image(points_grid, sigma, image_width, image_height):
    gaussians = np.full((image_height, image_width), fill_value=0, dtype=np.float64)
    for x, y in points_grid:
        gaussian = get_gaussian_on_image(x, y, sigma, image_width, image_height)
        gaussians = np.add(gaussians, gaussian)
    gaussians = np.clip(gaussians, a_min=0, a_max=1)
    return gaussians
    # TODO: this has basically become a visualization function - you can move it from here


def trunk_point_score(contours_mask, x, y, sigma):
    if not (0 <= x <= contours_mask.shape[1] and 0 <= y <= contours_mask.shape[0]):
        return 0 # TODO: think if I want to punish and give negative reward
    costs_mask = contours_mask.astype(np.int16)
    costs_mask[costs_mask == 0] = -255.0
    gaussian = get_gaussian_on_image(x, y, sigma, contours_mask.shape[1], contours_mask.shape[0])
    filter_result = np.multiply(gaussian, costs_mask)
    return np.sum(filter_result) # TODO: this returns nan from time to time


def trunks_grid_score(contours_mask, points_grid, sigma):
    return np.mean([trunk_point_score(contours_mask, x, y, sigma) for (x, y) in points_grid])


class TrunksGridOptimization(object):
    def __init__(self, grid_dim_x, grid_dim_y, translation, orientation, shear, sigma, image, n):
        self.init_grid_dim_x = grid_dim_x
        self.init_grid_dim_y = grid_dim_y
        self.init_translation_x = translation[0]
        self.init_translation_y = translation[1]
        self.init_orientation = orientation
        self.init_shear = shear
        self.init_sigma = sigma
        self.contours_mask = segmentation.extract_canopy_contours(image)[1]
        self.n = n
        self.width = image.shape[1]
        self.height = image.shape[0]

    def target(self, args):
        grid_dim_x, grid_dim_y, translation_x, translation_y, orientation, shear, sigma = args
        points_grid = get_grid(dim_x=grid_dim_x, dim_y=grid_dim_y, translation=(translation_x, translation_y), orientation=orientation, shear=shear, n=self.n)
        return trunks_grid_score(self.contours_mask, points_grid, sigma)

    def get_params(self):
        params = OrderedDict()
        params['grid_dim_x'] = ['integer', (max(0, self.init_grid_dim_x - 50), self.init_grid_dim_x + 50)] # TODO: take in percents of image dimensions (apply to all) or percents of init_drif_dim_x
        params['grid_dim_y'] = ['integer', (max(0, self.init_grid_dim_y - 50), self.init_grid_dim_y + 50)]
        params['translation_x'] = ['integer', (max(0, self.init_translation_x - 50), min(self.width, self.init_translation_x + 50))] # TODO: width?
        params['translation_y'] = ['integer', (max(0, self.init_translation_y - 50), min(self.height, self.init_translation_y + 50))] # TODO: height?
        params['orientation'] = ['real', (self.init_orientation - 5, self.init_orientation + 5)]
        params['shear'] = ['real', (self.init_shear - 0.1, self.init_shear + 0.1)]
        params['sigma'] = ['real', (max(0, self.init_sigma - 50), self.init_sigma + 50)]
        return params