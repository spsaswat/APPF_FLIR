import cv2
import numpy as np
from PIL import Image
from PIL import ImageEnhance
import json

# Load camera parameters, such as intrinsic matrix and distortion coefficients
def load_camera_parameters(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    camera_matrix = np.array(data['K'])
    dist_coeffs = np.array(data['dist'])
    return camera_matrix, dist_coeffs

# Function to undistort the image based on camera parameters
def undistort_image(img, camera_matrix, dist_coeffs):
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

# Image processing functions, including reading, undistorting, adjusting contrast, cropping, and rotating
def process_image(input_path, output_path, crop_coords, rotation_degrees=0, convert_gray=False, camera_matrix=None, dist_coeffs=None, contrast_factor=1.0):
    img = cv2.imread(input_path)
    if camera_matrix is not None and dist_coeffs is not None:
        img = undistort_image(img, camera_matrix, dist_coeffs)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if contrast_factor != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)
    cropped_img = img.crop(crop_coords)
    if rotation_degrees != 0:
        cropped_img = cropped_img.rotate(rotation_degrees, expand=True)
    if convert_gray:
        cropped_img = cropped_img.convert('L')
    cropped_img.save(output_path)
    return output_path

# Load camera parameters
camera_matrix1, dist_coeffs1 = load_camera_parameters("SensorCommunication/Acquisition/calib_data/test_plant_20240412161357/kdc_intrinsics.txt")
camera_matrix2, dist_coeffs2 = load_camera_parameters("SensorCommunication/Acquisition/calib_data/test_plant_20240412161357/kd_intrinsics.txt")

# Define paths and settings
input_img_path1 = "SensorCommunication/Acquisition/calib_data/test_plant_20240412161357/rgb_000001.png" # Cahnge File path !!
output_img_path1 = "SensorCommunication/Acquisition/calib_data/test_plant_20240412161357/processed_rgb_000001.png" # Cahnge File path !!
crop_coords1 = (600, 200, 820, 600)
rotation_degrees1 = -90
convert_gray1 = True

input_img_path2 = "SensorCommunication/Acquisition/calib_data/test_plant_20240412161357/DC_0010.jpg" # Cahnge File path and File name!!!
output_img_path2 = "SensorCommunication/Acquisition/calib_data/test_plant_20240412161357/processed_DC_0010.jpg" # Cahnge File path and File name!!!
crop_coords2 = (44, 135, 510, 385)
rotation_degrees2 = 0
convert_gray2 = True
contrast_factor = 1.8

# Process the image
processed_image_path1 = process_image(input_img_path1, output_img_path1, crop_coords1, rotation_degrees1, convert_gray1, camera_matrix1, dist_coeffs1, contrast_factor)
print(f"Processed first image with adjusted contrast saved at: {processed_image_path1}")
processed_image_path2 = process_image(input_img_path2, output_img_path2, crop_coords2, rotation_degrees2, convert_gray2, camera_matrix2, dist_coeffs2)
print(f"Processed second image saved at: {processed_image_path2}")

# Circle center matching and transformation matrix calculation
def find_circles(image_path, dp=0.1, minDist=30, param1=90, param2=5, minRadius=10, maxRadius=20):
# Read the image and convert to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
       
    # Apply Gaussian smoothing
    img_blurred = cv2.GaussianBlur(img, (5, 5),1)
    edges = img_blurred
    
    # Use HoughCircles to detect circles
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp, minDist,
                               param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        return [(circle[0], circle[1]) for circle in circles]
    else:
        return []

def match_circles(centers_rgb, centers_thermal):
    matched_rgb = []
    matched_thermal = []
    for ctr in centers_thermal:
        distances = [np.linalg.norm(np.array(ctr) - np.array(rgb_ctr)) for rgb_ctr in centers_rgb]
        min_index = np.argmin(distances)
        matched_rgb.append(centers_rgb[min_index])
        matched_thermal.append(ctr)
        centers_rgb.pop(min_index)
    return matched_rgb, matched_thermal

def compute_transformation(centers_src, centers_dst):
    if len(centers_src) < 3 or len(centers_dst) < 3:
        return None
    centers_src = np.float32(centers_src)
    centers_dst = np.float32(centers_dst)
    M, _ = cv2.findHomography(centers_src, centers_dst, cv2.RANSAC, 5.0)
    return M

# Circle center detection and matching
rgb_image_path = "SensorCommunication/Acquisition/calib_data/test_plant_20240412161357/processed_rgb_000001.png" # Cahnge File path!!!
thermal_image_path = "SensorCommunication/Acquisition/calib_data/test_plant_20240412161357/processed_DC_0010.jpg" # Cahnge File path and File name!!!
rgb_centers = find_circles(rgb_image_path)
thermal_centers = find_circles(thermal_image_path)
matched_rgb_centers, matched_thermal_centers = match_circles(rgb_centers, thermal_centers)

if matched_rgb_centers and matched_thermal_centers:
    transformation_matrix = compute_transformation(matched_rgb_centers, matched_thermal_centers)
    if transformation_matrix is not None:
        print("Transformation Matrix:")
        print(transformation_matrix)
        rgb_image = cv2.imread(rgb_image_path)
        thermal_image = cv2.imread(thermal_image_path, cv2.COLOR_BGR2RGB)
        aligned_rgb_image = cv2.warpPerspective(rgb_image, transformation_matrix, (thermal_image.shape[1], thermal_image.shape[0]))
        aligned_rgb_image_path = "SensorCommunication/Acquisition/calib_data/test_plant_20240412161357/aligned_rgb_image.png" # Cahnge File path!!!
        cv2.imwrite(aligned_rgb_image_path, aligned_rgb_image)
        print(f"Aligned RGB image saved at: {aligned_rgb_image_path}")
    else:
        print("Could not compute a valid transformation.")
else:
    print("Failed to match circles adequately.")