import numpy as np
from pykalman import KalmanFilter

class PointMassTracker(object):
    def __init__(self, frequency, transition_variances, observation_variances, init_pose, init_variances, change_thresholds):
        delta_t = 1.0 / frequency
        transition_mat = np.array([[1, 0, delta_t, 0],
                                   [0, 1, 0, delta_t],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])
        transition_cov = np.array([[transition_variances[0], 0, 0, 0],
                                   [0, transition_variances[1], 0, 0],
                                   [0, 0, transition_variances[2], 0],
                                   [0, 0, 0, transition_variances[3]]])
        observation_mat = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0]])
        observation_cov = np.array([[observation_variances[0], 0],
                                    [0, observation_variances[1]]])
        init_state = np.array([init_pose[0], init_pose[1], 0, 0])
        init_cov = np.array([[init_variances[0], 0, 0, 0],
                             [0, init_variances[1], 0, 0],
                             [0, 0, init_variances[2], 0],
                             [0, 0, 0, init_variances[3]]])
        self.change_thresholds = change_thresholds
        self.kf = KalmanFilter(transition_matrices=transition_mat, observation_matrices=observation_mat, transition_covariance=transition_cov,
                               observation_covariance=observation_cov, initial_state_mean=init_state, initial_state_covariance=init_cov)
        self.filtered_state_mean = init_state
        self.filtered_state_cov = init_cov

    def update_and_get_estimation(self, measurement):
        if measurement is not None:
            if abs(self.filtered_state_mean[0] - measurement[0]) > self.change_thresholds[0] or \
                    abs(self.filtered_state_mean[1] - measurement[1]) > self.change_thresholds[1]:
                measurement = None
        self.filtered_state_mean, self.filtered_state_cov = self.kf.filter_update(self.filtered_state_mean, self.filtered_state_cov, measurement)
        return self.filtered_state_mean