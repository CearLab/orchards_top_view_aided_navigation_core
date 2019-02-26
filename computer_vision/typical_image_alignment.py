import cv2
import numpy as np

from framework import cv_utils

def orb_based_registration(image, baseline_image, max_features=500, good_match_percent=0.15, transformation_type='affine'):

    # Convert images to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    baseline_image_gray = cv2.cvtColor(baseline_image, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create(max_features)
    keypoints, descriptors = orb.detectAndCompute(image_gray, None)
    keypoints_baseline, descriptors_baseline = orb.detectAndCompute(baseline_image_gray, None)

    # Match features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors, descriptors_baseline, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not-so-good matches
    good_matches_count = int(len(matches) * good_match_percent)
    matches = matches[:good_matches_count]

    # Get matches image
    matches_image = cv2.drawMatches(image_gray, keypoints, baseline_image_gray, keypoints_baseline, matches,
                                    outImg=None, matchColor=(255, 255, 0))

    # Extract location of good matches
    points = np.zeros((len(matches), 2), dtype=np.float32)
    points_baseline = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points[i, :] = keypoints[match.queryIdx].pt
        points_baseline[i, :] = keypoints_baseline[match.trainIdx].pt

    # Warp image
    registered_image, _ = cv_utils.warp_image(image, points, points_baseline, transformation_type)

    return registered_image, matches_image

