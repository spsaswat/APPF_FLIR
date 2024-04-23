import cv2
import numpy as np
import json

"""
Author: Zishuang Xing

Description: This script processes images from two different cameras (RGB and DC thermal) by applying geometric transformations
and corrections to align and calibrate them based on detected features. It includes distortion correction,
color-based feature extraction, circle detection, and perspective transformation to align images.

The pipeline loads camera intrinsic parameters, performs distortion correction, detects circles using Hough Transform, and calculates and applies a transformation matrix to align the images based on the
matched circle centers.
"""

def load_camera_parameters(file_path):
    """
    Load camera intrinsic parameters from a JSON file.

    Args:
    file_path (str): The path to the JSON file containing the camera parameters.

    Returns:
    tuple: A tuple containing the camera matrix (K) and distortion coefficients.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    K = np.array(data['K'])
    dist_coeffs = np.array(data['dist'])
    return K, dist_coeffs

def undistort_image(image, K, dist_coeffs):
    """
    Apply distortion correction to an image using the camera intrinsic parameters.

    Args:
    image (ndarray): The distorted image to be corrected.
    K (ndarray): The camera matrix.
    dist_coeffs (ndarray): The distortion coefficients.

    Returns:
    ndarray: The undistorted image.
    """
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(image, K, dist_coeffs, None, new_camera_matrix)
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]
    return undistorted_img

def create_mask_for_color_range(image, lower_color, upper_color):
    """
    Create a binary mask for a given color range in the HSV color space. This is often used for isolating
    specific colors within an image for further processing, such as contour detection.

    Args:
    image (ndarray): The image from which the mask will be created.
    lower_color (array): The lower bound of the HSV range.
    upper_color (array): The upper bound of the HSV range.

    Returns:
    ndarray: A binary mask.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv_image, lower_color, upper_color)
    return color_mask

def remove_outliers(data, max_diff=10):
    """
    Remove outliers from a dataset based on the difference threshold.

    Args:
    data (array): The dataset from which to remove outliers.
    max_diff (int): The maximum allowed difference between consecutive elements.

    Returns:
    array: The filtered dataset with outliers removed.
    """
    if len(data) < 2:
        #print("Not enough data to filter for outliers.")
        return data
    data = np.sort(data)
    diffs = np.abs(np.diff(data))  # Calculate differences between consecutive elements

    #print(f"Diffs: {diffs}")

    # Initialize a list to keep track of valid indices
    valid_indices = [0]  # Always include the first element
    
    # Iterate over the differences, starting from the second element
    for i in range(1, len(diffs) + 1):
        # Check the difference with the previous and the next data point
        if diffs[i-1] <= max_diff:
            valid_indices.append(i)
        elif i < len(diffs) and diffs[i] <= max_diff:
            # This ensures we are not skipping the point just after a single large jump if it's not an outlier
            valid_indices.append(i)
    
    # Extract the filtered data based on valid indices
    filtered_data = data[valid_indices]
    #print(f"Filtered Data: {filtered_data}")
    return filtered_data


def detect_circles(image, dp, minDist, param1, param2, minRadius, maxRadius, x_min, x_max, y_min, y_max, remove_outliers_flag=False):
    """
    Detect circles in an image using the Hough Transform algorithm. This function also allows for filtering
    circles based on spatial coordinates to limit the detection area within the image and optionally removes outliers.

    Args:
    image (ndarray): The image in which to detect circles.
    dp (float): Inverse ratio of the accumulator resolution to the image resolution.
    minDist (float): Minimum distance between the centers of the detected circles.
    param1 (float): The higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
    param2 (float): The accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected.
    minRadius (int): Minimum circle radius.
    maxRadius (int): Maximum circle radius.
    x_min, x_max, y_min, y_max (int): Spatial boundaries for circle detection.
    remove_outliers_flag (bool): If True, applies an outlier removal process to the circle centers based on their x-coordinates.

    Returns:
    list: A list of tuples, each representing the (x, y) coordinates and radius of a detected circle.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, minDist,
                               param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    centers = []
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        filtered_circles = [(x, y, r) for x, y, r in circles if x_min <= x <= x_max and y_min <= y <= y_max]
        if remove_outliers_flag:
            x_coords = np.array([x for x, _, _ in filtered_circles])
            filtered_x_coords = remove_outliers(x_coords)
            filtered_circles = [(x, y, r) for x, y, r in filtered_circles if x in filtered_x_coords]
        for x, y, r in filtered_circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(image, (x, y), 2, (0, 0, 255), 3)
            centers.append((x, y))
    return centers

# 加载相机参数
K_rgb, dist_coeffs_rgb = load_camera_parameters('SensorCommunication/Acquisition/calib_data/test_plant_20240412161903/kd_intrinsics.txt')
K_dc, dist_coeffs_dc = load_camera_parameters('SensorCommunication/Acquisition/calib_data/test_plant_20240412161903/kdc_intrinsics.txt')

# Process RGB image
rgb_image_path = 'SensorCommunication/Acquisition/calib_data/test_plant_20240412161903/rgb_000001.png'
rgb_image = cv2.imread(rgb_image_path)
rgb_image = undistort_image(rgb_image, K_rgb, dist_coeffs_rgb)
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])
blue_mask = create_mask_for_color_range(rgb_image, lower_blue, upper_blue)
contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
box_contour = max(contours, key=cv2.contourArea) if contours else None

if box_contour is not None:
    mask = np.zeros_like(rgb_image)
    cv2.drawContours(mask, [box_contour], 0, (255, 255, 255), -1)
    masked_rgb_image = cv2.bitwise_and(rgb_image, mask)
else:
    masked_rgb_image = rgb_image
    #print("No blue box contour found.")

blurred_rgb_image = cv2.GaussianBlur(masked_rgb_image, (5, 5), 0)
rgb_circle_centers = detect_circles(blurred_rgb_image, dp=1, minDist=30, param1=50, param2=25, minRadius=10, maxRadius=30, x_min=590, x_max=810, y_min=0, y_max=600, remove_outliers_flag=True)
cv2.imwrite('SensorCommunication/Acquisition/calib_data/test_plant_20240412161903/detected_rgb.png', blurred_rgb_image)

# Process DC image
dc_image_path = 'SensorCommunication/Acquisition/calib_data/test_plant_20240412161903/DC_0016.jpg'
dc_image = cv2.imread(dc_image_path, cv2.IMREAD_GRAYSCALE)
dc_image = undistort_image(dc_image, K_dc, dist_coeffs_dc)
dc_circle_centers = detect_circles(dc_image, dp=1, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=30, x_min=0, x_max=1000, y_min=120, y_max=400, remove_outliers_flag=False)
cv2.imwrite('SensorCommunication/Acquisition/calib_data/test_plant_20240412161903/detected_dc.png', dc_image)

# Convert list of centers to NumPy arrays for further processing
rgb_circle_centers_np = np.array(rgb_circle_centers)
dc_circle_centers_np = np.array(dc_circle_centers)
#print("RGB Circle Centers:", rgb_circle_centers_np)
#print("DC Circle Centers:", dc_circle_centers_np)

def sort_centers(centers):
    """
    Sorts the centers of detected features (e.g., circles) within an image. Sorting is primarily based on the y-coordinate
    to maintain a top-to-bottom order, and secondarily on the x-coordinate for a left-to-right order within the same horizontal level
    
    Args:
    centers (ndarray): An array of circle centers to sort.

    Returns:
    ndarray: The sorted array of centers, which facilitates the establishment of one-to-one correspondences between features in different images
    """
    sorted_indices = np.lexsort((centers[:,0], centers[:,1]))  # x first, y second, because y is the primary sort key
    return centers[sorted_indices]

# Sort the detected centers for both RGB and DC images.
sorted_rgb_centers = sort_centers(rgb_circle_centers_np)
sorted_dc_centers = sort_centers(dc_circle_centers_np)

def sort_matrix_rows_by_x(matrix):
    """
    Sorts each row of a matrix of circle centers by their x-coordinates. This is particularly useful when
    the circle centers need to be aligned or compared row-wise.

    Args:
    matrix (ndarray): A 2D array where each row contains circle centers.

    Returns:
    ndarray: The matrix with each row sorted by x-coordinates.
    """
    for i in range(matrix.shape[0]):
        matrix[i] = matrix[i][np.argsort(matrix[i][:, 0])]
    return matrix


# Arrange the centers in a matrix form and then sort each row by the x-coordinates.
rgb_matrix = sorted_rgb_centers.reshape(4, 3, 2)  # RGB matrix 4x3
dc_matrix = sorted_dc_centers.reshape(3, 4, 2)    # DC matrix 3x4

sorted_rgb_matrix = sort_matrix_rows_by_x(rgb_matrix.copy())
sorted_dc_matrix = sort_matrix_rows_by_x(dc_matrix.copy())

#print("Sorted RGB Circle Centers Matrix:")
#print(sorted_rgb_matrix)
#print("\nSorted DC Circle Centers Matrix:")
#print(sorted_dc_matrix)


def compute_transformation_matrix(rgb_centers, dc_centers, matches):
    """
    Compute a transformation matrix from source points to destination points using homography.

    Args:
    rgb_centers (ndarray): Array of circle centers in the RGB image.
    dc_centers (ndarray): Array of circle centers in the DC image.
    matches (dict): A dictionary mapping indices from DC centers to corresponding RGB centers.

    Returns:
    ndarray: The computed transformation matrix.
    """
    # Extract corresponding points from given matches
    src_pts = np.float32([dc_centers[i-1] for i in matches.keys()]).reshape(-1, 1, 2)
    dst_pts = np.float32([rgb_centers[i-1] for i in matches.values()]).reshape(-1, 1, 2)

    # Calculate transformation matrix
    matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return matrix

def apply_transformation(src_image, matrix, dimensions):
    """
    Apply a perspective transformation to an image.

    Args:
    src_image (ndarray): The source image to transform.
    matrix (ndarray): The transformation matrix.
    dimensions (tuple): The dimensions (width, height) of the output image.

    Returns:
    ndarray: The transformed image.
    """
    transformed_image = cv2.warpPerspective(src_image, matrix, dimensions)
    return transformed_image

# Define correspondences between detected circles in DC and RGB images for transformation.
matches = {
    1: 10, 2: 7, 3: 4, 4: 1,
    5: 11, 6: 8, 7: 5, 8: 2,
    9: 12, 10: 9, 11: 6, 12: 3
}

# Compute the transformation matrix from DC to RGB using the defined matches.
transformation_matrix = compute_transformation_matrix(sorted_rgb_matrix.reshape(-1, 2), sorted_dc_matrix.reshape(-1, 2), matches)
print("Transformation Matrix:\n", transformation_matrix)
# Apply the computed transformation to align the DC image with the RGB image.
aligned_dc_image = apply_transformation(dc_image, transformation_matrix, (rgb_image.shape[1], rgb_image.shape[0]))
cv2.imwrite('SensorCommunication/Acquisition/calib_data/test_plant_20240412161903/aligned_dc_image.png', aligned_dc_image)
