import cv2
import numpy as np

import experiments_framework.framework.viz_utils as viz_utils


def register(image, base_image, max_features=500, good_match_percent=0.15, show_matches=False): # TODO: verify the order!!!

    # Convert images to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    base_image_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create(max_features)
    keypoints, descriptors = orb.detectAndCompute(image_gray, None)
    keypoints_base, descriptors_base = orb.detectAndCompute(base_image_gray, None)

    # Match features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors, descriptors_base, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    good_matches_count = int(len(matches) * good_match_percent)
    matches = matches[:good_matches_count]

    if show_matches:
        matches_image = cv2.drawMatches(image, keypoints, base_image, keypoints_base, matches, None)
        viz_utils.show_image('matches', matches_image)

    # Extract location of good matches
    points = np.zeros((len(matches), 2), dtype=np.float32)
    points_base = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points[i, :] = keypoints[match.queryIdx].pt
        points_base[i, :] = keypoints_base[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points, points_base, cv2.RANSAC)

    # Use homography
    height, width, channels = base_image.shape
    registered_image = cv2.warpPerspective(image, h, (width, height))

    return registered_image, h