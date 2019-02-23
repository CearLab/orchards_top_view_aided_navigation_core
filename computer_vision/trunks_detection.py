import itertools
from collections import OrderedDict
import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from computer_vision import segmentation
from framework import cv_utils
from nelder_mead import NelderMead


def estimate_rows_orientation(image, search_step=0.5, min_distance_between_peaks=200, min_peak_width=50):
    _, canopies_mask = segmentation.extract_canopy_contours(image)
    angle_to_score = {}
    angle_to_sum_vector = {}
    for correction_angle in np.arange(start=-90, stop=90, step=search_step):
        rotation_mat = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), correction_angle, scale=1.0)
        rotated_canopies_mask = cv2.warpAffine(canopies_mask, rotation_mat, (canopies_mask.shape[1], canopies_mask.shape[0]))
        column_sums_vector = np.sum(rotated_canopies_mask, axis=0)
        angle_to_sum_vector[correction_angle] = column_sums_vector
        minima_indices, _ = find_peaks(column_sums_vector * (-1), distance=min_distance_between_peaks, width=min_peak_width)
        minima_values = [column_sums_vector[index] for index in minima_indices]
        minima_mean = np.mean(minima_values) if len(minima_values) > 0 else 1e30
        angle_to_score[correction_angle] = minima_mean
    orientation = [key for key, value in sorted(angle_to_score.iteritems(), key=lambda (k, v): v, reverse=False)][0] * (-1)
    return orientation, angle_to_score, angle_to_sum_vector


def find_tree_centroids(image, correction_angle):
    rotation_mat = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), correction_angle, scale=1.0)
    rotated_image = cv2.warpAffine(image, rotation_mat, (image.shape[1], image.shape[0]))
    rotated_centroids = []
    _, canopies_mask = segmentation.extract_canopy_contours(rotated_image)
    column_sums_vector = np.sum(canopies_mask, axis=0)
    aisle_centers, _ = find_peaks(column_sums_vector * (-1), distance=200, width=50)
    slices_sum_vectors_and_trees = []
    for tree_row_left_limit, tree_tow_right_limit in zip(aisle_centers[:-1], aisle_centers[1:]):
        tree_row = canopies_mask[:, tree_row_left_limit:tree_tow_right_limit]
        row_sums_vector = np.sum(tree_row, axis=1)
        tree_locations_in_row, _ = find_peaks(row_sums_vector, distance=160, width=30)
        rotated_centroids.append([(int(np.mean([tree_row_left_limit, tree_tow_right_limit])), tree_location) for tree_location in tree_locations_in_row])
        slices_sum_vectors_and_trees.append((tree_row, row_sums_vector, tree_locations_in_row))
    vertical_rows_centroids_np = np.float32(list(itertools.chain.from_iterable(rotated_centroids))).reshape(-1, 1, 2)
    rotation_mat = np.insert(cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), correction_angle * (-1), scale=1.0), [2], [0, 0, 1], axis=0)
    centroids_np = cv2.perspectiveTransform(vertical_rows_centroids_np, rotation_mat)
    centroids = [tuple(elem) for elem in centroids_np[:, 0, :].tolist()]
    return centroids, rotated_centroids, aisle_centers, slices_sum_vectors_and_trees, column_sums_vector


def estimate_grid_dimensions(rotated_centroids):
    row_locations = map(lambda row: np.median([centroid[0] for centroid in row]), rotated_centroids)
    dim_x = np.median(np.array(row_locations[1:]) - np.array(row_locations[:-1]))
    tree_distances = []
    for row in rotated_centroids:
        tree_locations = sorted([centroid[1] for centroid in row])
        tree_distances += list(np.array(tree_locations[1:]) - np.array(tree_locations[:-1]))
    dim_y = np.percentile(tree_distances, q=30)
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
    thetas_filtered = reject_outliers(thetas, m=1.5)
    drift_vectors_filtered = [drift_vectors[index] for index in [thetas.index(theta) for theta in thetas_filtered]]
    shear = np.sin(np.median(thetas_filtered))
    return shear, drift_vectors, drift_vectors_filtered


def get_essential_grid(dim_x, dim_y, shear, orientation, n, m=None):
    if m is None:
        m = n
    nodes = list(itertools.product(np.arange(0, n * dim_x, step=dim_x), np.arange(0, m * dim_y, step=dim_y)))
    nodes_np = np.float32(nodes).reshape(-1, 1, 2)
    shear_mat = np.float32([[1, 0, 0], [shear, 1, 0], [0, 0, 1]])
    transformed_nodes_np = cv2.perspectiveTransform(nodes_np, shear_mat)
    rotation_center = tuple(np.mean(transformed_nodes_np, axis=0)[0])
    rotation_mat = np.insert(cv2.getRotationMatrix2D(rotation_center, orientation, scale=1.0), [2], [0, 0, 1], axis=0)
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
        if np.any(candidate_grid_np < 0) or np.any(candidate_grid_np[:,0] >= image_height) or np.any(candidate_grid_np[:,1] >= image_width):
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


def get_grid(dim_x, dim_y, translation, orientation, shear, n, m=None):
    essential_grid = get_essential_grid(dim_x, dim_y, shear, orientation, n, m)
    essential_grid_np = np.float32(essential_grid).reshape(-1, 1, 2)
    translation_mat = np.float32([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])
    transformed_grid_np = cv2.perspectiveTransform(essential_grid_np, translation_mat)
    transformed_grid = [tuple(elem) for elem in transformed_grid_np[:, 0, :].tolist()]
    return transformed_grid


def get_gaussian_on_image(mu_x, mu_y, sigma, image_width, image_height, square_size_to_sigma_ratio=2, circle_radius_to_sigma_ratio=1.5):
    gaussian_image = np.full((image_height, image_width), fill_value=0, dtype=np.float64)
    square_size = square_size_to_sigma_ratio * sigma
    circle_radius = circle_radius_to_sigma_ratio * sigma
    x_start, x_end = max(0, int(mu_x - square_size)), min(image_width, int(mu_x + square_size))
    y_start, y_end = max(0, int(mu_y - square_size)), min(image_height, int(mu_y + square_size))
    x, y = np.meshgrid(np.arange(x_start, x_end), np.arange(y_start, y_end))
    squre_gaussian = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2.0 * sigma ** 2))
    circle_mask = cv2.circle(img=np.full(squre_gaussian.shape, fill_value=0.0, dtype=np.float64),
                             center=(int(mu_x - x_start), int(mu_y - y_start)), radius=int(circle_radius), color=1.0, thickness=-1)
    circle_gaussian = np.multiply(squre_gaussian, circle_mask)
    gaussian_image = cv_utils.insert_image_patch(gaussian_image, circle_gaussian, upper_left=(x_start, y_start), lower_right=(x_end, y_end))
    return gaussian_image


def get_gaussians_grid_image(points_grid, sigma, image_width, image_height, scale_factor=1.0, square_size_to_sigma_ratio=2, circle_radius_to_sigma_ratio=1.5):
    gaussians = np.full((image_height, image_width), fill_value=0, dtype=np.float64)
    for x, y in points_grid:
        gaussian = scale_factor * get_gaussian_on_image(x, y, sigma, image_width, image_height,
                                                        square_size_to_sigma_ratio, circle_radius_to_sigma_ratio)
        gaussians = np.maximum(gaussians, gaussian)
    gaussians = np.clip(gaussians, a_min=0, a_max=1)
    return gaussians


def tree_score(canopies_mask, x, y, sigma):
    if not (0 <= x <= canopies_mask.shape[1] and 0 <= y <= canopies_mask.shape[0]):
        return -np.inf, -1
    reward_mask = canopies_mask.astype(np.int16)
    reward_mask[reward_mask == 0] = -255.0
    gaussian = get_gaussian_on_image(x, y, sigma, canopies_mask.shape[1], canopies_mask.shape[0])
    filter_result = np.multiply(gaussian, reward_mask)
    score = np.sum(filter_result)
    normalized_score = score / (255.0 * np.sum(gaussian))
    return score, normalized_score


def get_tree_scores_stats(canopies_mask, points_grid, sigma):
    tree_scores = []
    normalized_tree_scores = []
    for (x, y) in points_grid:
        score, normalized_score = tree_score(canopies_mask, x, y, sigma)
        tree_scores.append(score)
        normalized_tree_scores.append(normalized_score)
    stats = {'mean_score': np.mean(tree_scores),
             'std_score': np.std(tree_scores),
             'median_score': np.median(tree_scores),
             'mean_normalized_score': np.mean(normalized_tree_scores),
             'std_normalized_score': np.std(normalized_tree_scores),
             'median_normalized_score': np.median(normalized_tree_scores)}
    return stats


def get_trees_confidence(canopies_mask, trunk_coordinates_np, no_trunk_coordinates_np, sigma):
    trunk_points_list = filter(lambda v: v==v, trunk_coordinates_np)
    no_trunk_points_list = filter(lambda v: v==v, no_trunk_coordinates_np)
    return np.mean([tree_score(canopies_mask, x, y, sigma)[1] for (x, y) in trunk_points_list] +
                   [tree_score(canopies_mask, x, y, sigma)[1] * (-1) for (x, y) in no_trunk_points_list])


class _TrunksGridOptimization(object):
    def __init__(self, grid_dim_x, grid_dim_y, translation, orientation, shear, sigma, image, n, m, pattern,
                 std_normalized_tree_scores_threshold=0.6):
        self.init_grid_dim_x = grid_dim_x
        self.init_grid_dim_y = grid_dim_y
        self.init_translation_x = translation[0]
        self.init_translation_y = translation[1]
        self.init_orientation = orientation
        self.init_shear = shear
        self.init_sigma = sigma
        self.canopies_mask = segmentation.extract_canopy_contours(image)[1]
        self.n = n
        self.m = m
        self.pattern = pattern
        self.std_normalized_tree_scores_threshold = std_normalized_tree_scores_threshold
        self.steps = []
        self.width = image.shape[1]
        self.height = image.shape[0]

    def target(self, args):

        # Get list of trunk points
        grid_dim_x, grid_dim_y, translation_x, translation_y, orientation, shear, sigma = args
        points_grid = get_grid(dim_x=grid_dim_x, dim_y=grid_dim_y, translation=(translation_x, translation_y), orientation=orientation, shear=shear, n=self.n, m=self.m)
        points_grid_np = np.empty(len(points_grid), dtype=object)
        points_grid_np[:] = points_grid
        points_grid_np = points_grid_np.reshape(self.pattern.shape, order='F')
        trunk_points_np = points_grid_np[self.pattern != -1]
        trunk_points = trunk_points_np.tolist()

        # Calculate score per tree
        tree_scores = []
        normalized_tree_scores = []
        for (x, y) in trunk_points:
            score, normalized_score = tree_score(self.canopies_mask, x, y, sigma)
            tree_scores.append(score)
            normalized_tree_scores.append(normalized_score)

        # Calculate pattern score
        std_normalized_score = np.std(normalized_tree_scores)
        if std_normalized_score > self.std_normalized_tree_scores_threshold:
            return -np.inf
        pattern_score = np.mean(tree_scores)
        self.steps.append((points_grid, pattern_score, sigma))
        return pattern_score

    def get_params(self, dims_margin=60, translation_margin=60, orientation_margin=7, shear_margin=0.12, sigma_margin=50, initial_volume_factor=0.2):
        params = OrderedDict()
        params['grid_dim_x'] = ['integer', (max(0, self.init_grid_dim_x - dims_margin), self.init_grid_dim_x + dims_margin)]
        params['grid_dim_y'] = ['integer', (max(0, self.init_grid_dim_y - dims_margin), self.init_grid_dim_y + dims_margin)]
        params['translation_x'] = ['integer', (self.init_translation_x - translation_margin, min(self.width, self.init_translation_x + translation_margin))]
        params['translation_y'] = ['integer', (self.init_translation_y - translation_margin, min(self.height, self.init_translation_y + translation_margin))]
        params['orientation'] = ['real', (self.init_orientation - orientation_margin, self.init_orientation + orientation_margin)]
        params['shear'] = ['real', (self.init_shear - shear_margin, self.init_shear + shear_margin)]
        params['sigma'] = ['real', (max(0, self.init_sigma - sigma_margin), self.init_sigma + sigma_margin)]

        initial_simplex = [[self.init_grid_dim_x, self.init_grid_dim_y, self.init_translation_x, self.init_translation_y,
                            self.init_orientation, self.init_shear, self.init_sigma] for _ in range(8)]
        initial_simplex[0][0] += int(initial_volume_factor * dims_margin)
        initial_simplex[1][1] += int(initial_volume_factor * dims_margin)
        initial_simplex[2][2] += int(initial_volume_factor * translation_margin)
        initial_simplex[3][3] += int(initial_volume_factor * translation_margin)
        initial_simplex[4][4] += initial_volume_factor * orientation_margin
        initial_simplex[5][5] += initial_volume_factor * shear_margin
        initial_simplex[6][6] += initial_volume_factor * sigma_margin
        initial_simplex[7][0] -= int(initial_volume_factor * dims_margin)
        initial_simplex[7][0] = max(0, initial_simplex[7][0])
        initial_simplex[7][1] -= int(initial_volume_factor * dims_margin)
        initial_simplex[7][1] = max(0, initial_simplex[7][1])
        initial_simplex[7][2] -= int(initial_volume_factor * translation_margin)
        initial_simplex[7][3] -= int(initial_volume_factor * translation_margin)
        initial_simplex[7][4] -= initial_volume_factor * orientation_margin
        initial_simplex[7][5] -= initial_volume_factor * shear_margin
        return params, initial_simplex


def optimize_grid(grid_dim_x, grid_dim_y, translation, orientation, shear, sigma, cropped_image, pattern, iterations_num=30):
    opt = _TrunksGridOptimization(grid_dim_x, grid_dim_y, translation, orientation, shear, sigma, cropped_image, pattern.shape[1], pattern.shape[0], pattern)
    params, initial_simplex = opt.get_params()
    nm = NelderMead(opt.target, params, verbose=True)
    nm.initialize(initial_simplex)
    optimized_grid_args, _ = nm.maximize(n_iter=iterations_num)
    optimized_grid_dim_x, optimized_grid_dim_y, optimized_translation_x, optimized_translation_y, optimized_orientation, optimized_shear, optimized_sigma = optimized_grid_args
    optimized_grid = get_grid(optimized_grid_dim_x, optimized_grid_dim_y, (optimized_translation_x, optimized_translation_y), optimized_orientation, optimized_shear, pattern.shape[1], pattern.shape[0])
    return optimized_grid, optimized_grid_args, opt.steps


def extrapolate_full_grid(grid_dim_x, grid_dim_y, orientation, shear, base_grid_origin, image_width, image_height):
    shear_angle = np.arcsin(shear)
    n = int(np.max([2 * image_width / grid_dim_x, 2 * image_height / grid_dim_y]))
    full_grid = get_essential_grid(grid_dim_x, grid_dim_y, shear, orientation, n)
    full_grid = np.array(full_grid) - np.array(full_grid[0]) + base_grid_origin
    rotation_pivot = (full_grid[0][0], full_grid[0][1])
    rotation_mat = np.insert(cv2.getRotationMatrix2D(rotation_pivot, (-1) * orientation, scale=1.0), [2], [0, 0, 1], axis=0)
    full_grid_np = full_grid.reshape(-1, 1, 2)
    full_grid_rotated_np = cv2.perspectiveTransform(full_grid_np, rotation_mat)
    full_grid_rotated = [tuple(elem) for elem in full_grid_rotated_np[:, 0, :].tolist()]
    full_grid_rotated_and_shifted = (np.array(full_grid_rotated) -
                                     np.array([int(n / 3) * grid_dim_x, int(n / 3) * (grid_dim_x * np.tan(shear_angle) + grid_dim_y)])).reshape(-1, 1, 2)
    rotation_mat = np.insert(cv2.getRotationMatrix2D(rotation_pivot, orientation, scale=1.0), [2], [0, 0, 1], axis=0)
    full_grid_shifted_np = cv2.perspectiveTransform(full_grid_rotated_and_shifted, rotation_mat)
    full_grid_shifted = [tuple(elem) for elem in full_grid_shifted_np[:, 0, :].tolist()]
    full_grid_df = pd.DataFrame(index=range(n), columns=range(n))
    for i in range(n):
        for j in range(n):
            coordinates = full_grid_shifted[i + n * j]
            full_grid_df.loc[i, j] = coordinates if (0 <= coordinates[0] < image_width and 0 <= coordinates[1] < image_height) else np.nan
    full_grid_df = full_grid_df.dropna(axis=0, how='all').dropna(axis=1, how='all')
    full_grid_np = np.array(full_grid_df)
    return full_grid_np


def get_grid_scores_array(full_grid_np, image, sigma):
    _, canopies_mask = segmentation.extract_canopy_contours(image)
    full_grid_scores_np = np.empty(full_grid_np.shape)
    full_grid_pose_to_score = {}
    for i in range(full_grid_np.shape[0]):
        for j in range((full_grid_np.shape[1])):
            if np.any(np.isnan(full_grid_np[(i, j)])):
                full_grid_scores_np[(i, j)] = np.nan
            else:
                x, y = full_grid_np[(i, j)]
                score = tree_score(canopies_mask, x, y, sigma)[1]
                full_grid_scores_np[(i, j)] = score
                full_grid_pose_to_score[(int(x), int(y))] = score
    return full_grid_scores_np, full_grid_pose_to_score


def fit_pattern_on_grid(scores_array_np, pattern_np):
    max_mean_scores = -np.inf
    maximizing_origin = None
    origin_to_sub_scores_array = {}
    for i in range(scores_array_np.shape[0]):
        for j in range(scores_array_np.shape[1]):
            if i + pattern_np.shape[0] > scores_array_np.shape[0] or j + pattern_np.shape[1] > scores_array_np.shape[1]:
                continue
            sub_scores_array_np = scores_array_np[i : i + pattern_np.shape[0], j : j + pattern_np.shape[1]]
            if not np.all(np.logical_or(pattern_np != 1, np.logical_and(np.bitwise_not(np.isnan(sub_scores_array_np)), pattern_np == 1))):
                continue
            mean_score = np.mean(np.multiply(np.nan_to_num(sub_scores_array_np), pattern_np))
            origin_to_sub_scores_array[(i, j)] = np.multiply(np.nan_to_num(sub_scores_array_np), pattern_np)
            if mean_score > max_mean_scores:
                max_mean_scores = mean_score
                maximizing_origin = (i, j)
    return maximizing_origin, origin_to_sub_scores_array


def refine_trunk_locations(image, trunk_coordinates_np, sigma, dim_x, dim_y, samples_along_axis=14):
    _, canopies_mask = segmentation.extract_canopy_contours(image)
    refined_trunk_locations_df = pd.DataFrame(index=range(trunk_coordinates_np.shape[0]), columns=range(trunk_coordinates_np.shape[1]))
    window_size = int(np.max([dim_x, dim_y]) * 1.1)
    window_shift = int(sigma / 3)
    for i in range(trunk_coordinates_np.shape[0]):
        for j in range(trunk_coordinates_np.shape[1]):
            if np.any(np.isnan(trunk_coordinates_np[(i, j)])):
                continue
            x, y = trunk_coordinates_np[(i, j)]
            max_score = -np.inf
            best_x, best_y = None, None
            for candidate_x, candidate_y in itertools.product(np.round(np.linspace(x - window_shift, x + window_shift, num=samples_along_axis)),
                                                              np.round(np.linspace(y - window_shift, y + window_shift, num=samples_along_axis))):
                canopy_patch, _, _ = cv_utils.crop_region(canopies_mask, candidate_x, candidate_y, window_size, window_size)
                score, _ = tree_score(canopy_patch, canopy_patch.shape[1] / 2, canopy_patch.shape[0] / 2, sigma)
                if score > max_score:
                    max_score = score
                    best_x, best_y = candidate_x, candidate_y
            refined_trunk_locations_df.loc[i, j] = (best_x, best_y)
    return np.array(refined_trunk_locations_df)
