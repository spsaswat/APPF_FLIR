import os
import cv2
import numpy as np
import json

def load_camera_parameters(file_path):
    """Load camera intrinsic parameters from a JSON-like text file."""
    with open(file_path, 'r') as file:
        data = json.load(file)

    K = np.array(data['K'])
    dist = np.array(data['dist'])
    return K, dist

def undistort_image(image, K, dist):
    """Undistort an image using the camera intrinsic parameters."""
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(image, K, dist, None, new_camera_matrix)
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]
    return undistorted_img

def apply_transformation(image, matrix, output_shape):
    """Apply a perspective transformation to an image."""
    return cv2.warpPerspective(image, matrix, output_shape, flags=cv2.INTER_NEAREST)

def convert_pixel_to_temperature(pixel_value, calibration_params):
    gain = calibration_params['gain']
    offset = calibration_params['offset']
    temperature = gain * pixel_value + offset
    return temperature

def is_valid_transformation_matrix(matrix):
    """Check if the transformation matrix is valid."""
    if matrix is None or matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        print("Invalid transformation matrix detected.")
        return False
    return True

def transform_leaf_masks(leaf_mask_path, transformation_matrix, output_shape):
    """Transform leaf masks to align with the thermal image."""
    leaf_masks = np.load(leaf_mask_path)
    transformed_masks = []

    if not is_valid_transformation_matrix(transformation_matrix):
        print("Skipping transformation due to invalid matrix.")
        return [None] * leaf_masks.shape[0]  # Return a list of None of the same length as leaf_masks

    for i in range(leaf_masks.shape[0]):
        leaf_mask_image = (leaf_masks[i].astype(np.uint8) * 255)
        try:
            transformed_mask = cv2.warpPerspective(leaf_mask_image, transformation_matrix, output_shape, flags=cv2.INTER_NEAREST)
            if transformed_mask.shape[:2] != output_shape:
                print(f"Resizing mask {i} from {transformed_mask.shape} to {output_shape}.")
                transformed_mask = cv2.resize(transformed_mask, output_shape, interpolation=cv2.INTER_NEAREST)
            transformed_mask = transformed_mask.astype(bool)
            transformed_masks.append(transformed_mask)
        except cv2.error as e:
            print(f"OpenCV error during warping of mask {i}: {e}")
            transformed_masks.append(None)
    return transformed_masks

def extract_leaf_temperatures(transformed_masks, dc_image, calibration_params):
    """Extract temperatures for each leaf from the thermal image."""
    leaf_temperatures = []
    for i, mask in enumerate(transformed_masks):
        if mask is None:
            print(f"Skipping mask {i} as it is None.")
            leaf_temperatures.append(None)
            continue
        
        if mask.shape != dc_image.shape:
            print(f"Mask size {mask.shape} does not match image size {dc_image.shape}, resizing mask.")
            mask = cv2.resize(mask.astype(np.uint8), (dc_image.shape[1], dc_image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        pixel_values = dc_image[mask]
        temperatures = convert_pixel_to_temperature(pixel_values, calibration_params)
        leaf_temperatures.append(temperatures)

    return leaf_temperatures

def save_temperatures(leaf_temperatures, filename):
    """Save the extracted temperatures to a file."""
    try:
        # Convert to an object dtype array or save using np.save with allow_pickle=True
        np.save(filename, np.array(leaf_temperatures, dtype=object), allow_pickle=True)
    except Exception as e:
        raise ValueError(f"Error saving temperatures: {e}")

def main():
    # Define the directories based on your setup
    leaf_mask_dir = os.path.normpath('../SensorCommunication/Acquisition/batch_1/test_plant_20240903103507')
    thermal_image_dir = os.path.normpath('../SensorCommunication/Acquisition/batch_1/test_plant_20240903103507')
    camera_params_dir = os.path.normpath('../SensorCommunication/Acquisition/batch_1/test_plant_20240903102437')

    rgb_camera_params_path = os.path.join(camera_params_dir, 'kd_intrinsics.txt')
    thermal_camera_params_path = os.path.join(camera_params_dir, 'kdc_intrinsics.txt')

    calibration_params = {'gain': 0.04, 'offset': -273.15}

    # Load camera parameters
    K_dc, dist_dc = load_camera_parameters(thermal_camera_params_path)
    K_rgb, dist_rgb = load_camera_parameters(rgb_camera_params_path)

    # Process each thermal image and its corresponding mask
    for thermal_image_name in os.listdir(thermal_image_dir):
        if thermal_image_name.endswith('.tiff') or thermal_image_name.endswith('.tif'):
            thermal_image_path = os.path.join(thermal_image_dir, thermal_image_name)
            dc_image = cv2.imread(thermal_image_path, cv2.IMREAD_UNCHANGED)

            # Ensure the thermal image is loaded correctly
            if dc_image is None:
                print(f"Failed to load the thermal image from {thermal_image_path}.")
                continue
            
            # Check the format of the thermal image
            if dc_image.dtype != np.uint16:
                print(f"The thermal image should be in 16-bit mono format, but got {dc_image.dtype}. Skipping {thermal_image_name}.")
                continue

            # Derive the corresponding mask file name
            base_name = os.path.splitext(thermal_image_name)[0]
            mask_name = f"{base_name.replace('DC-', 'rgb_')}_mask.npy"
            mask_path = os.path.join(leaf_mask_dir, mask_name)
            print(f"Looking for mask file: {mask_path}")

            if not os.path.exists(mask_path):
                print(f"No corresponding leaf mask file found for {thermal_image_name}. Skipping.")
                continue

            # Load transformation matrix if available
            transformation_matrix_path = os.path.join(camera_params_dir, f"transformation_matrix{base_name.replace('DC-', '_')}.npy")
            print(f"Looking for transformation matrix: {transformation_matrix_path}")
            
            if not os.path.exists(transformation_matrix_path):
                print(f"No transformation matrix found for {thermal_image_name}. Skipping.")
                continue

            transformation_matrix = np.load(transformation_matrix_path)
            
            # Get the output shape based on the thermal image size
            output_shape = (dc_image.shape[1], dc_image.shape[0])

            # Undistort the thermal image
            undistorted_dc_image = undistort_image(dc_image, K_dc, dist_dc)

            # Transform leaf masks to match the thermal image
            transformed_masks = transform_leaf_masks(mask_path, transformation_matrix, output_shape)

            # Extract temperatures for each leaf from the thermal image
            leaf_temperatures = extract_leaf_temperatures(transformed_masks, undistorted_dc_image, calibration_params)

            # Print the extracted temperatures for each leaf
            for i, temperatures in enumerate(leaf_temperatures):
                if temperatures is not None:
                    print(f"Leaf {i}:")
                    print(f"  Min temperature: {np.min(temperatures):.2f}°C")
                    print(f"  Max temperature: {np.max(temperatures):.2f}°C")
                    print(f"  Mean temperature: {np.mean(temperatures):.2f}°C")
                else:
                    print(f"No temperatures extracted for leaf {i}.")

            # Save the extracted temperatures to a file
            save_temperatures(leaf_temperatures, filename=f'{base_name}_temperatures.npy')

if __name__ == "__main__":
    main()
