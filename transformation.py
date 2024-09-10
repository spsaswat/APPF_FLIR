import cv2
import numpy as np
import os
import re

def load_transformation_matrix(folder_path, suffix):
    """
    Load the transformation matrix from a file.

    Args:
    folder_path (str): The folder containing the transformation matrices.
    suffix (str): The unique identifier for the image.

    Returns:
    ndarray: The transformation matrix.
    """
    matrix_path = os.path.join(folder_path, f"transformation_matrix_{suffix}.npy")
    print(f"Loading transformation matrix from: {matrix_path}")
    if not os.path.exists(matrix_path):
        print(f"Transformation matrix file does not exist: {matrix_path}")
        return None

    try:
        matrix = np.load(matrix_path)
        print(f"Loaded transformation matrix: {matrix}")
        return matrix
    except Exception as e:
        print(f"Error loading transformation matrix: {e}")
        return None

def apply_transformation(image, matrix, dimensions):
    """
    Apply a transformation matrix to an image.

    Args:
    image (ndarray): Input image to be transformed.
    matrix (ndarray): Transformation matrix.
    dimensions (tuple): Desired output dimensions.

    Returns:
    ndarray: Transformed image.
    """
    print(f"Applying transformation matrix to the image with dimensions: {dimensions}")
    try:
        transformed_image = cv2.warpPerspective(image, matrix, dimensions)
        print("Transformation applied successfully.")
        return transformed_image
    except Exception as e:
        print(f"Error applying transformation: {e}")
        return image

def adjust_rgb_to_dc_visible_area(rgb_image_path, dc_image_path):
    """
    Adjust the RGB image to match the visible area of the DC image.

    Args:
    rgb_image (ndarray): The RGB image.
    dc_image (ndarray): The DC image.

    Returns:
    ndarray: The adjusted RGB image.
    """
    # Load the RGB and DC images
    rgb_image = cv2.imread(rgb_image_path)
    dc_image = cv2.imread(dc_image_path, cv2.IMREAD_GRAYSCALE)

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
    
    return black_mask


def apply_transformations_on_images(image_folder, transformation_folder, output_folder):
    """
    Apply transformation matrices to the corresponding images in another folder and save aligned images.

    Args:
    image_folder (str): The folder containing the images (RGB and DC).
    transformation_folder (str): The folder containing the transformation matrices.
    output_folder (str): The folder to save the transformed images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(image_folder):
        match = re.match(r'(rgb|DC)[-_](\d+)', file_name)
        if match:
            image_type, suffix = match.groups()
            image_path = os.path.join(image_folder, file_name)

            print(f"Processing image: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
            print(suffix)

            # Load corresponding RGB image to get its dimensions
            rgb_image_path = os.path.join(image_folder, f"rgb_{suffix}.png")  # Assuming RGB images are .png format
            rgb_image = cv2.imread(rgb_image_path)
            if rgb_image is None:
                print(f"Failed to load corresponding RGB image: {rgb_image_path}")
                continue
            # Use the dimensions of the RGB image for both DC and RGB transformations
            dimensions = (rgb_image.shape[1], rgb_image.shape[0])

            transformation_matrix = load_transformation_matrix(transformation_folder, suffix)
            if transformation_matrix is None:
                continue

            #dimensions = (image.shape[1], image.shape[0])
            # Extract the original file extension
            _, ext = os.path.splitext(file_name)
            if ext == '.png':
                continue
            elif ext == '.tiff':
                aligned_image = apply_transformation(image, transformation_matrix, dimensions)
            
            aligned_image_path = os.path.join(output_folder, f'aligned_detected_{image_type}_{suffix}{ext}')
            # Save the image with the same file extension as the original file
            cv2.imwrite(aligned_image_path, aligned_image)
            print(f"Saved aligned image to: {aligned_image_path}")
            
            rgb_image_path = os.path.join(output_folder, f'rgb_{suffix}.png')
            dc_image_path = aligned_image_path
            # Adjust the RGB image to match the visible area of the DC image
            aligned_rgb_image = adjust_rgb_to_dc_visible_area(rgb_image_path, dc_image_path)
            # Save the adjusted RGB image
            aligned_rgb_image_path = os.path.join(output_folder, f'aligned_detected_rgb_{suffix}.png')
            cv2.imwrite(aligned_rgb_image_path, aligned_rgb_image)
            print(f"Saved adjusted RGB image to: {aligned_rgb_image_path}")



if __name__ == "__main__":
    image_folder = "SensorCommunication/Acquisition/batch_1/test_plant_20240903103507/"
    transformation_folder = "SensorCommunication/Acquisition/batch_1/test/"
    output_folder = image_folder

    apply_transformations_on_images(image_folder, transformation_folder, output_folder)
