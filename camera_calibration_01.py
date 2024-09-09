import cv2
import numpy as np
import json
import os

def load_camera_parameters(file_path):
    """Load camera intrinsic parameters from a JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    K = np.array(data['K'])
    dist_coeffs = np.array(data['dist'])
    return K, dist_coeffs

def undistort_image(image, K, dist_coeffs):
    """Apply distortion correction to an image using the camera intrinsic parameters."""
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(image, K, dist_coeffs, None, new_camera_matrix)
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]
    return undistorted_img

def apply_x_coordinate_mask(image, x_min=435, x_max=960):
    """Apply a mask to an image, blacking out regions outside the specified x-coordinate range."""
    mask = np.zeros_like(image)
    mask[:, x_min:x_max] = 255
    return cv2.bitwise_and(image, mask)

def remove_outliers(data, max_diff=10):
    """Remove outliers from a dataset based on the difference threshold."""
    if len(data) < 2:
        return data
    data = np.sort(data)
    diffs = np.abs(np.diff(data))
    valid_indices = [0]
    for i in range(1, len(diffs) + 1):
        if diffs[i-1] <= max_diff or (i < len(diffs) and diffs[i] <= max_diff):
            valid_indices.append(i)
    return data[valid_indices]

def preprocess_for_rgb_circle_detection(image):
    """Preprocess RGB images for circle detection."""
    # Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Apply Gaussian blur to reduce noise and improve circle detection
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Normalize to 8-bit format for Hough Circle detection
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return normalized

def preprocess_for_dc_circle_detection(image):
    """Preprocess DC images for circle detection."""
    # Apply any DC-specific preprocessing like denoising or sharpening
    filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Convert to 8-bit format
    normalized = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return normalized

def detect_circles(image, dp, minDist, param1, param2, minRadius, maxRadius, x_min, x_max, y_min, y_max, remove_outliers_flag=False, is_dc_image=False):
    """Detect circles in an image using the Hough Transform algorithm."""
    # Apply preprocessing specifically for RGB or DC images
    if is_dc_image:
        processed_image = preprocess_for_dc_circle_detection(image)
    else:
        processed_image = preprocess_for_rgb_circle_detection(image)

    # Apply Hough Circle Detection
    circles = cv2.HoughCircles(processed_image, cv2.HOUGH_GRADIENT, dp, minDist,
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

def apply_transformation(src_image, matrix, dimensions):
    """Apply a perspective transformation to an image."""
    transformed_image = cv2.warpPerspective(src_image, matrix, dimensions)
    return transformed_image

def adjust_rgb_to_dc_visible_area(rgb_path, dc_path, output_path):
    """Adjust the RGB image to match the visible area of the DC image."""
    rgb_image = cv2.imread(rgb_path)
    dc_image = cv2.imread(dc_path, cv2.IMREAD_GRAYSCALE)

    _, dc_mask = cv2.threshold(dc_image, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(dc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    visible_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(visible_contour)
    black_mask = np.zeros_like(rgb_image)
    black_mask[y:y+h, x:x+w] = rgb_image[y:y+h, x:x+w]
    cv2.imwrite(output_path, black_mask)

def compute_transformation_matrix(rgb_centers, dc_centers):
    """Compute a transformation matrix from source points to destination points using homography."""
    num_points = min(len(rgb_centers), len(dc_centers))

    # Ensure there are at least four points to compute homography
    if num_points < 4:
        print(f"Not enough points to compute the transformation matrix. Points available: {num_points}")
        return None

    # Prepare the points for homography calculation
    src_pts = np.float32(dc_centers[:num_points]).reshape(-1, 1, 2)
    dst_pts = np.float32(rgb_centers[:num_points]).reshape(-1, 1, 2)

    # Calculate the transformation matrix
    matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return matrix

def process_images(base_path, folder_name):
    """Main processing function to handle the alignment of RGB and DC images based on predefined workflows."""
    folder_path = os.path.join(base_path, folder_name)
    K_rgb, dist_coeffs_rgb = load_camera_parameters(os.path.join(folder_path, 'kd_intrinsics.txt'))
    K_dc, dist_coeffs_dc = load_camera_parameters(os.path.join(folder_path, 'kdc_intrinsics.txt'))

    rgb_circle_centers_dict = {}
    dc_circle_centers_dict = {}
    detected_rgb_filenames = []
    detected_dc_filenames = []
    transformation_matrix_dict = {}

    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)
        if file_name.startswith('rgb') and file_name.endswith('.png'):
            rgb_suffix = file_name.split('_')[-1].split('.')[0]
            rgb_image = cv2.imread(image_path)
            rgb_image = undistort_image(rgb_image, K_rgb, dist_coeffs_rgb)
            masked_image = apply_x_coordinate_mask(rgb_image)
            rgb_circle_centers = detect_circles(masked_image, dp=1.2, minDist=40, param1=50, param2=30, minRadius=5, maxRadius=25, x_min=620, x_max=850, y_min=200, y_max=530, remove_outliers_flag=True)
            rgb_circle_centers_dict[rgb_suffix] = rgb_circle_centers
            detected_rgb_filename = 'detected_' + file_name
            detected_rgb_filenames.append(detected_rgb_filename)
            cv2.imwrite(os.path.join(folder_path, detected_rgb_filename), masked_image)

        elif file_name.startswith('DC') and file_name.endswith('.tiff'):
            dc_suffix = file_name.split('-')[-1].split('.')[0]
            dc_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            dc_image = undistort_image(dc_image, K_dc, dist_coeffs_dc)
            dc_circle_centers = detect_circles(dc_image, dp=1.2, minDist=40, param1=50, param2=28, minRadius=5, maxRadius=25, x_min=120, x_max=360, y_min=100, y_max=500, remove_outliers_flag=False, is_dc_image=True)
            dc_circle_centers_dict[dc_suffix] = dc_circle_centers
            detected_dc_filename = 'detected_' + file_name
            detected_dc_filenames.append(detected_dc_filename)
            cv2.imwrite(os.path.join(folder_path, detected_dc_filename), dc_image)

    for suffix in rgb_circle_centers_dict.keys():  
        if suffix in dc_circle_centers_dict:
            rgb_circle_centers = rgb_circle_centers_dict[suffix]
            dc_circle_centers = dc_circle_centers_dict[suffix]

            if rgb_circle_centers and dc_circle_centers:
                sorted_rgb_centers = np.array(rgb_circle_centers)
                sorted_dc_centers = np.array(dc_circle_centers)

                # Check if there are enough points to compute the transformation matrix
                transformation_matrix = compute_transformation_matrix(sorted_rgb_centers, sorted_dc_centers)
                if transformation_matrix is not None:
                    transformation_matrix_dict[suffix] = transformation_matrix
                    matrix_file_path = os.path.join(folder_path, f'transformation_matrix_{suffix}.npy')
                    np.save(matrix_file_path, transformation_matrix)
                    print(f"Transformation matrix saved to: {matrix_file_path}")
                else:
                    print(f"Skipped saving transformation matrix for {suffix} due to insufficient points.")

    for detected_dc_filename, detected_rgb_filename in zip(detected_dc_filenames, detected_rgb_filenames):
        dc_image_path = os.path.join(folder_path, detected_dc_filename)
        dc_image = cv2.imread(dc_image_path, cv2.IMREAD_GRAYSCALE)
        suffix = detected_rgb_filename.split('_')[-1].split('.')[0]

        # Ensure the transformation matrix exists before applying it
        if suffix in transformation_matrix_dict and transformation_matrix_dict[suffix] is not None:
            aligned_dc_image = apply_transformation(dc_image, transformation_matrix_dict[suffix], (masked_image.shape[1], masked_image.shape[0]))
            aligned_dc_image_path = os.path.join(folder_path, 'aligned_' + detected_dc_filename)
            cv2.imwrite(aligned_dc_image_path, aligned_dc_image)

            aligned_dc_circles = detect_circles(aligned_dc_image, dp=1.2, minDist=40, param1=50, param2=30, minRadius=5, maxRadius=25, x_min=590, x_max=810, y_min=0, y_max=1000, remove_outliers_flag=False, is_dc_image=True)
            adjust_rgb_to_dc_visible_area(os.path.join(folder_path, detected_rgb_filename), aligned_dc_image_path, os.path.join(folder_path, 'aligned_' + detected_rgb_filename))
        else:
            print(f"Transformation matrix for {suffix} is missing or insufficient points; skipping alignment.")

if __name__ == "__main__":
    base_path = 'SensorCommunication/Acquisition/batch_1'
    print("Enter the folder name (e.g., test_plant_20240412161903):")
    folder_name = input()
    process_images(base_path, folder_name)
