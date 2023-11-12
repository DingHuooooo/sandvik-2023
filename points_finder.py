import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import erosion, disk
from scipy.ndimage import label, center_of_mass
from skimage.measure import find_contours, points_in_poly
from skimage.morphology import erosion, disk
from skimage.draw import line

def bisect_region(mask, centroid, direction_vector):
    """Bisects the mask along the direction_vector starting from the centroid."""
    new_mask = mask.copy()
    
    end_point1 = centroid + 1000 * direction_vector
    end_point2 = centroid - 1000 * direction_vector
    rr, cc = line(int(end_point1[0]), int(end_point1[1]), int(end_point2[0]), int(end_point2[1]))
    
    valid_indices = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
    rr = rr[valid_indices]
    cc = cc[valid_indices]

    # Set to 0 (cut) along the line
    new_mask[rr, cc] = 0
    return new_mask

def closest_boundary_point(mask, point):
    # 1. Identify the contours of the mask.
    contours = find_contours(mask, 0.5)
    
    # 2. If no contours are found, simply return the given point.
    if len(contours) == 0:
        return point
    
    # 3. Sort the contours based on their length and pick the longest one.
    longest_contour = sorted(contours, key=lambda x: len(x))[-1]

    # 4. For each point on the contour, calculate its distance from the given point.
    distances = np.linalg.norm(longest_contour - point, axis=1)
    
    # 5. Select the contour point that has the shortest distance to the given point.
    nearest_boundary_point = longest_contour[np.argmin(distances)]

    return nearest_boundary_point

def is_on_boundary(point, contour):
    """Check if a point is on a contour."""
    return any(np.all(point == contour_pt) for contour_pt in contour)

def check_and_adjust_centroids(mask, depth=0, max_depth=3):
    """Check and adjust centroids recursively with a maximum depth."""
    
    # Base case: if we reach the maximum depth, terminate the recursion
    if depth >= max_depth:
        return []
    
    labeled_mask, num_features = label(mask)
    centroids = []

    for region_label in range(1, num_features + 1):
        region_mask = (labeled_mask == region_label)
        gravity_centroid = center_of_mass(region_mask)

        # Visualization
        contours = find_contours(region_mask, 0.1)[0]

        y, x = int(gravity_centroid[0]), int(gravity_centroid[1])
        is_inside = region_mask[y, x]
        on_boundary = is_on_boundary(gravity_centroid, contours)

        if not is_inside:
            if on_boundary:
                # Find another boundary point and compute the perpendicular direction
                nearest_boundary_point = closest_boundary_point(region_mask, gravity_centroid)
                direction_vector = nearest_boundary_point - gravity_centroid
                perpendicular_vector = np.array([-direction_vector[1], direction_vector[0]])
                cut_mask = bisect_region(region_mask, gravity_centroid, perpendicular_vector)
            else:
                nearest_boundary_point = closest_boundary_point(region_mask, gravity_centroid)
                direction_vector = nearest_boundary_point - gravity_centroid
                cut_mask = bisect_region(region_mask, gravity_centroid, direction_vector)
            centroids += check_and_adjust_centroids(cut_mask, depth + 1)
        else:
            centroids.append(gravity_centroid)

    return centroids

# The function will now recursively adjust centroids up to a maximum depth of 3.



def find_centroids_gravity_center_recursive(pre_mask, ratio=0.05):
    labeled_mask, num_features = label(pre_mask)
    area_sizes = np.bincount(labeled_mask.ravel())[1:]
    total_mask_area = np.sum(pre_mask)
    area_threshold = max(ratio * total_mask_area, 500)
    large_region_indices = np.where(area_sizes > area_threshold)[0] + 1

    large_region_centroids = []
    large_region_labels = []
    for region_label in large_region_indices:
        region_mask = (labeled_mask == region_label)
        current_centroids = check_and_adjust_centroids(region_mask)
        large_region_centroids += current_centroids
        large_region_labels += [region_label] * (len(current_centroids))

    return labeled_mask, large_region_labels, large_region_centroids