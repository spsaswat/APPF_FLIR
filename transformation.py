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
            transformation_matrix = load_transformation_matrix(transformation_folder, suffix)
            if transformation_matrix is None:
                continue

            dimensions = (image.shape[1], image.shape[0])
            aligned_image = apply_transformation(image, transformation_matrix, dimensions)

            aligned_image_path = os.path.join(output_folder, f'aligned_detected_{image_type}_{suffix}.tiff')
            cv2.imwrite(aligned_image_path, aligned_image)
            print(f"Saved aligned image to: {aligned_image_path}")

# Run the function
if __name__ == "__main__":
    image_folder = "SensorCommunication/Acquisition/batch_1/test/"
    transformation_folder = "SensorCommunication/Acquisition/batch_1/test/"
    output_folder = image_folder

    apply_transformations_on_images(image_folder, transformation_folder, output_folder)
