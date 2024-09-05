import cv2
import numpy as np
import json
import os
import re

"""
Author: Zishuang Xing

Description: This script processes images from two different cameras (RGB and DC thermal) by applying geometric transformations
and corrections to align and calibrate them based on detected features. It includes distortion correction,
color-based feature extraction, circle detection, and perspective transformation to align images.

The core functionalities include:
- Loading camera intrinsic parameters from JSON files.
- Applying distortion correction to the images.
- Detecting circular features within the images.
- Computing and applying a perspective transformation based on matched features between the two image types.
- Adjusting the RGB image to only include the area visible in the corresponding DC image.
"""

import json
import cv2
import numpy as np

def load_camera_parameters(file_path):
    """
    Load camera intrinsic parameters from a JSON file.

    Args:
    file_path (str): The path to the JSON file containing the camera parameters.

    Returns:
    tuple: A tuple containing the camera matrix (K) and distortion coefficients.
    """
    print(f"Loading camera parameters from: {file_path}")
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        K = np.array(data['K'])
        dist_coeffs = np.array(data['dist'])
        print(f"Loaded camera matrix: {K}")
        print(f"Loaded distortion coefficients: {dist_coeffs}")
        return K, dist_coeffs
    except Exception as e:
        print(f"Error loading camera parameters: {e}")
        return None, None


def rotate_image_90_degrees(image_path, output_path):
    """
    Rotates a 16-bit image by 90 degrees clockwise and saves the result.

    Args:
    image_path (str): The path to the input image file.
    output_path (str): The path to save the rotated image.
    """
    # Load the image in unchanged mode to preserve the 16-bit depth
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Check if the image is loaded correctly
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

    # Rotate the image 90 degrees clockwise
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    # Save the rotated image
    cv2.imwrite(output_path, rotated_image)

def undistort_image(image, K, dist_coeffs):
    """
    Apply distortion correction to an image.

    Args:
    image (ndarray): Input image to be undistorted.
    K (ndarray): Camera matrix.
    dist_coeffs (ndarray): Distortion coefficients.

    Returns:
    ndarray: Undistorted image.
    """
    print("Starting image undistortion...")
    if K is None or dist_coeffs is None:
        print("Error: Camera parameters are missing.")
        return image

    try:
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))
        undistorted_image = cv2.undistort(image, K, dist_coeffs, None, new_camera_matrix)
        print("Undistortion completed.")
        return undistorted_image
    except Exception as e:
        print(f"Error during undistortion: {e}")
        return image

def apply_x_coordinate_mask(image, x_min=435, x_max=960):
    """
    Apply a mask to an image, blacking out regions outside the specified x-coordinate range.

    Args:
    image (ndarray): The source image to mask.
    x_min (int): The minimum x-coordinate for the visible region.
    x_max (int): The maximum x-coordinate for the visible region.

    Returns:
    ndarray: The masked image.
    """
    print(f"Applying mask from x={x_min} to x={x_max}")
    mask = np.zeros_like(image)
    mask[:, x_min:x_max] = 255
    return cv2.bitwise_and(image, mask)

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
        return data
    data = np.sort(data)
    diffs = np.abs(np.diff(data))
    valid_indices = [0]  # Always include the first element
    
    # Iterate over the differences, starting from the second element
    for i in range(1, len(diffs) + 1):
        if diffs[i-1] <= max_diff:
            valid_indices.append(i)
        elif i < len(diffs) and diffs[i] <= max_diff:
            valid_indices.append(i)
    return data[valid_indices]

def detect_circles(image, dp, minDist, param1, param2, minRadius, maxRadius, x_min, x_max, y_min, y_max, remove_outliers_flag=False):
    """
    Detect circles in an image using the Hough Transform algorithm. This function also allows for filtering
    circles based on spatial coordinates to limit the detection area within the image and optionally removes outliers.

    Args:
    image (ndarray): The image in which to detect circles.
    dp (float): Inverse ratio of the accumulator resolution to the image resolution.
    minDist (float): Minimum distance between the centers of the detected circles.
    param1 (float): The higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
    param2 (float): The accumulator threshold for the circle centers at the detection stage.
    minRadius (int): Minimum circle radius.
    maxRadius (int): Maximum circle radius.
    x_min, x_max, y_min, y_max (int): Spatial boundaries for circle detection.
    remove_outliers_flag (bool): If True, applies an outlier removal process to the circle centers based on their x-coordinates.

    Returns:
    list: A list of tuples, each representing the (x, y) coordinates and radius of a detected circle.
    """
    print("Detecting circles in the image...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, minDist,
                               param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    centers = []
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        filtered_circles = [(x, y, r) for x, y, r in circles if x_min <= x <= x_max and y_min <= y <= y_max]
        print(f"Detected {len(filtered_circles)} circles within specified range.")
        if remove_outliers_flag:
            x_coords = np.array([x for x, _, _ in filtered_circles])
            filtered_x_coords = remove_outliers(x_coords)
            filtered_circles = [(x, y, r) for x, y, r in filtered_circles if x in filtered_x_coords]
            print(f"Filtered to {len(filtered_circles)} circles after removing outliers.")
        centers = [(x, y) for x, y, _ in filtered_circles]
    else:
        print("No circles detected.")
    return centers

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

def compute_transformation_matrix(rgb_centers, dc_centers):
    """
    Compute a transformation matrix from source points to destination points using homography.

    Args:
    rgb_centers (ndarray): Array of circle centers in the RGB image.
    dc_centers (ndarray): Array of circle centers in the DC image.

    Returns:
    ndarray: The computed transformation matrix.
    """
    print("Computing transformation matrix...")
    num_points = min(len(rgb_centers), len(dc_centers))
    if num_points == 0:
        print("No points available to compute the transformation matrix.")
        return None
    src_pts = dc_centers[:num_points].reshape(-1, 1, 2)
    dst_pts = rgb_centers[:num_points].reshape(-1, 1, 2)
    matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print(f"Transformation matrix computed: {matrix}")
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

def calculate_alignment_error(sorted_rgb_centers, sorted_dc_centers):
    """
    Calculate the alignment error between sorted RGB and DC circle centers using Euclidean distance.
    
    Parameters:
        sorted_rgb_centers (list of tuples): Sorted list of RGB circle centers.
        sorted_dc_centers (list of tuples): Sorted list of DC circle centers.
        
    Returns:
        float: Average alignment error in pixels.
    """
    if len(sorted_rgb_centers) != len(sorted_dc_centers):
        raise ValueError("The number of centers in both lists must be the same to calculate alignment error.")

    errors = []
    for (x1, y1), (x2, y2) in zip(sorted_rgb_centers, sorted_dc_centers):
        dx = np.float32(x2) - np.float32(x1)
        dy = np.float32(y2) - np.float32(y1)
        errors.append(np.sqrt(dx**2 + dy**2))
    return np.mean(errors)

def adjust_rgb_to_dc_visible_area(rgb_path, dc_path, output_path):
    """
    Adjust the RGB image to match the visible area of the DC image.

    Args:
    rgb_path (str): Path to the RGB image.
    dc_path (str): Path to the DC image.
    output_path (str): Path to save the adjusted RGB image.
    """
    # Load the RGB and DC images
    rgb_image = cv2.imread(rgb_path)
    dc_image = cv2.imread(dc_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the DC image to create a mask of the visible area
    _, dc_mask = cv2.threshold(dc_image, 1, 255, cv2.THRESH_BINARY)
    # Find contours in the mask
    contours, _ = cv2.findContours(dc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Assume the largest contour is the visible area
    visible_contour = max(contours, key=cv2.contourArea)
    # Get the bounding rectangle of the visible contour
    x, y, w, h = cv2.boundingRect(visible_contour)
    # Create a black mask with the same dimensions as the RGB image
    black_mask = np.zeros_like(rgb_image)
    # Define the visible area on the black mask based on the bounding rectangle
    black_mask[y:y+h, x:x+w] = rgb_image[y:y+h, x:x+w]
    # Save the adjusted RGB image with the black mask applied
    cv2.imwrite(output_path, black_mask)

def process_folder_for_transformation(base_path, folder_name):
    """
    Processes images from a specified folder to align RGB and DC images using feature detection and geometric transformations.

    Args:
    base_path (str): The base directory where different sets of image data are stored.
    folder_name (str): The specific folder within the base directory that contains the image sets to be processed.

    This function loads camera parameters, undistorts images, detects circular features, computes transformations,
    applies transformations to align images, and calculates the alignment error between transformed RGB and DC images.
    """
    # Construct the full path to the folder containing images
    folder_path = os.path.join(base_path, folder_name)
    print(f"Processing folder: {folder_path}")

    # Load camera intrinsic parameters for RGB and DC cameras
    K_rgb, dist_coeffs_rgb = load_camera_parameters(os.path.join(folder_path, 'kd_intrinsics.txt'))
    K_dc, dist_coeffs_dc = load_camera_parameters(os.path.join(folder_path, 'kdc_intrinsics.txt'))

    # Dictionary to store pairs of RGB and DC images by matching suffixes in filenames
    image_pairs = {}

    # Iterate through the files in the folder to find matching RGB and DC image pairs
    for file_name in os.listdir(folder_path):
        print(f"Found file: {file_name}")
        match = re.match(r'(rgb|DC)[-_](\d+)', file_name)  # Updated regex to match both underscores and hyphens
        if match:
            image_type, suffix = match.groups()
            if suffix not in image_pairs:
                image_pairs[suffix] = {}  # Initialize dictionary for storing matched image pairs
            image_pairs[suffix][image_type] = file_name  # Store image filenames by their type (RGB or DC)

    # Process each pair of RGB and DC images
    for suffix, images in image_pairs.items():
        if 'rgb' in images and 'DC' in images:  # Ensure both RGB and DC images are available for this suffix
            print(f"Processing image pair: RGB: {images['rgb']}, DC: {images['DC']}")

            # Construct full paths for RGB and DC images
            rgb_image_path = os.path.join(folder_path, images['rgb'])
            dc_image_path = os.path.join(folder_path, images['DC'])

            # Load RGB and DC images
            rgb_image = cv2.imread(rgb_image_path)
            dc_image = cv2.imread(dc_image_path, cv2.IMREAD_GRAYSCALE)

            # Undistort images using their respective camera parameters
            rgb_image = undistort_image(rgb_image, K_rgb, dist_coeffs_rgb)
            masked_rgb = apply_x_coordinate_mask(rgb_image)  # Apply mask to RGB image to restrict detection area
            dc_image = undistort_image(dc_image, K_dc, dist_coeffs_dc)

            # Detect circles in RGB and DC images
            rgb_centers = detect_circles(masked_rgb, dp=1, minDist=30, param1=50, param2=25, minRadius=10, maxRadius=30,
                                         x_min=620, x_max=850, y_min=200, y_max=530, remove_outliers_flag=True)
            dc_centers = detect_circles(dc_image, dp=1, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=30,
                                        x_min=120, x_max=360, y_min=100, y_max=500, remove_outliers_flag=False)

            # Proceed only if both RGB and DC images have detected circles
            if rgb_centers and dc_centers:
                print(f"Detected {len(rgb_centers)} circles in RGB and {len(dc_centers)} circles in DC.")
                # Sort the detected circle centers for proper alignment
                sorted_rgb_centers = np.array(rgb_centers)
                sorted_dc_centers = np.array(dc_centers)

                # Compute the transformation matrix using detected circle centers
                transformation_matrix = compute_transformation_matrix(sorted_rgb_centers, sorted_dc_centers)

                if transformation_matrix is not None:
                    # Save the transformation matrix to a file for potential reuse
                    matrix_file_path = os.path.join(folder_path, f'transformation_matrix_{suffix}.npy')
                    np.save(matrix_file_path, transformation_matrix)
                    print(f"Transformation matrix saved to: {matrix_file_path}")
                else:
                    print("Failed to compute the transformation matrix; skipping saving.")

if __name__ == "__main__":
    base_path = 'SensorCommunication/Acquisition/batch_1'
    print("Enter the folder name (e.g., test_plant_20240412161903):")
    folder_name = input()
    process_folder_for_transformation(base_path, folder_name)

