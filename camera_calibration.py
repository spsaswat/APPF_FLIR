import cv2
import numpy as np
import json
import os

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
    matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
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

def process_images(base_path, folder_name):
    """
    Main processing function to handle the alignment of RGB and DC images based on predefined workflows. It performs
    the following sequence of operations: loading camera parameters, undistorting images, detecting features,
    and applying transformations to align the images.

    Args:
    base_path (str): The base directory where different sets of image data are stored.
    folder_name (str): The specific folder within the base directory that contains the image sets to be processed.

    The function processes each image found in the specified directory, applies geometric transformations,
    and saves the transformed images back to disk. It assumes that the folder structure and naming convention of files 
    are consistent with the expected input types (RGB and DC images).

    This function orchestrates the workflow by calling other specialized functions for image processing tasks, ensuring
    that images are processed in an order that respects dependencies between operations (e.g., distortion correction before feature detection).
    """
        
    folder_path = os.path.join(base_path, folder_name)
    K_rgb, dist_coeffs_rgb = load_camera_parameters(os.path.join(folder_path, 'kd_intrinsics.txt'))
    K_dc, dist_coeffs_dc = load_camera_parameters(os.path.join(folder_path, 'kdc_intrinsics.txt'))

    # Variables to store circle centers from different images
    rgb_circle_centers = None
    dc_circle_centers = None

    # Process RGB and DC images, and store detected images' names
    detected_rgb_filename = None
    detected_dc_filenames = []

    # Process each image file within the directory
    for file_name in os.listdir(folder_path):
        if file_name.startswith('rgb') and file_name.endswith('.png'):
            # Load and process RGB images
            image_path = os.path.join(folder_path, file_name)
            image = cv2.imread(image_path)
            image = undistort_image(image, K_rgb, dist_coeffs_rgb)
            masked_image = apply_x_coordinate_mask(image)
            rgb_circle_centers = detect_circles(masked_image, dp=1, minDist=30, param1=50, param2=25, minRadius=10, maxRadius=30, x_min=590, x_max=810, y_min=0, y_max=600, remove_outliers_flag=True)
            detected_rgb_filename = 'detected_' + file_name
            cv2.imwrite(os.path.join(folder_path, detected_rgb_filename), masked_image)

        elif file_name.startswith('DC') and file_name.endswith('.jpg'):
            # Load and process DC images
            image_path = os.path.join(folder_path, file_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = undistort_image(image, K_dc, dist_coeffs_dc)
            dc_circle_centers = detect_circles(image, dp=1, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=30, x_min=0, x_max=1000, y_min=120, y_max=400, remove_outliers_flag=False)
            detected_dc_filename = 'detected_' + file_name
            detected_dc_filenames.append(detected_dc_filename)
            cv2.imwrite(os.path.join(folder_path, detected_dc_filename), image)
            print(dc_circle_centers)

    # Process for transformation and alignment
    if rgb_circle_centers is not None and dc_circle_centers is not None:
        sorted_rgb_centers = sort_centers(np.array(rgb_circle_centers))
        sorted_rgb_matrix = sort_matrix_rows_by_x(sorted_rgb_centers.reshape(4, 3, 2))

        sorted_dc_centers = sort_centers(np.array(dc_circle_centers))
        sorted_dc_matrix = sort_matrix_rows_by_x(sorted_dc_centers.reshape(3, 4, 2))

        # Define correspondences and compute transformation matrix
        matches = {
            1: 10, 2: 7, 3: 4, 4: 1,
            5: 11, 6: 8, 7: 5, 8: 2,
            9: 12, 10: 9, 11: 6, 12: 3
        }
        # Match features between RGB and DC images and compute the transformation matrix
        transformation_matrix = compute_transformation_matrix(sorted_rgb_matrix.reshape(-1, 2), sorted_dc_matrix.reshape(-1, 2), matches)
        print("Transformation Matrix:\n", transformation_matrix)
        
        # Apply the computed transformation to align the DC images
        for detected_dc_filename in detected_dc_filenames:
            dc_image_path = os.path.join(folder_path, detected_dc_filename)
            dc_image = cv2.imread(dc_image_path, cv2.IMREAD_GRAYSCALE)
            aligned_dc_image = apply_transformation(dc_image, transformation_matrix, (masked_image.shape[1], masked_image.shape[0]))
            aligned_dc_image_path = os.path.join(folder_path, 'aligned_' + detected_dc_filename)
            cv2.imwrite(aligned_dc_image_path, aligned_dc_image)

            # Adjust RGB image to match the visible area of the DC image
            if detected_rgb_filename:
                detected_rgb_image_path = os.path.join(folder_path, detected_rgb_filename)
                adjusted_rgb_image_path = os.path.join(folder_path, 'aligned_' + detected_rgb_filename)
                adjust_rgb_to_dc_visible_area(detected_rgb_image_path, aligned_dc_image_path, adjusted_rgb_image_path)


if __name__ == "__main__":
    base_path = 'SensorCommunication/Acquisition/calib_data'
    #folder_name ='test_plant_20240412161357'
    print("Enter the folder name (e.g., test_plant_20240412161357):")
    folder_name = input()  # Get folder name from user input
    process_images(base_path, folder_name)
