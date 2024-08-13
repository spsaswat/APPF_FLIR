# Guide to Setting Up and Running the Camera Calibration Code
This guide provides step-by-step instructions for running the camera calibration code. These instructions are meant for users who have access to a terminal and are familiar with basic command line operations.

## 1. Purpose of the Calibration Code

This calibration code is designed to fine-tune and align images captured from both RGB and thermal cameras. The main objective is to ensure that these images are perfectly aligned, enhancing the accuracy and effectiveness of further image processing tasks like segmentation and analysis in plant phenotyping. 

## 2. Algorithm and Calibration Principles
This calibration script employs image processing technologies and geometric transformation methods to ensure that images captured from different cameras are precisely aligned and analyzed with high quality.
### 2.1 Reading Camera Parameters
- **Parameter Retrieval**:
The script initially imports the internal camera parameters from a pre-set JSON file, including the camera matrix (K) and distortion coefficients (dist_coeffs). The camera matrix describes the camera’s internal optical characteristics, while the distortion coefficients detail the lens’s radial and tangential distortions, foundational for accurate image calibration.

### 2.2 Image Distortion Correction
- **Distortion Correction**:
Using the cv2.undistort function from the OpenCV library, the script removes distortions from the images based on the extracted camera parameters. By computing a new camera matrix and remapping the image, this step eliminates the effects of distortion, producing a distortion-free, clear image. This is crucial for ensuring the precision of subsequent image alignments.

### 2.3 Mask Application
- **Purpose of the Mask**:
To exclude non-target areas in the image (such as slider structures), a mask is used to cover these areas, ensuring they do not interfere with the detection of feature points and the accuracy of image alignment.
- **Mask Setup**:
y specifying concrete values for x and y coordinates, the mask covers areas outside the slider, preserving the central area containing key features. This technique ensures that image processing is focused on areas with important information, thereby enhancing the efficiency and accuracy of the process.

### 2.4 Circle Detection
- **Circle Detection Method**:
The script employs the Hough Circle Transform, a feature extraction technique used to detect circles in images. This method applies a Hough Transform algorithm specifically designed to recognize circular shapes by identifying the accumulation of points in the image that form a circle. By adjusting parameters such as the resolution of the accumulator (dp), minimum distance between circles (minDist), and thresholds for the Canny edge detector (param1) and center detection (param2), the script efficiently finds circles within specified ranges of radii.
- **Feature Point Extraction**:
After detecting circles, the script extracts the centers of these circles as feature points. These feature points are then used in the homography matrix calculation to align images accurately.

### 2.5 Outlier Removal in Circle Detection
- **Purpose of Outlier Removal**:
The script includes a method named `remove_outliers` to refine the feature points detected during the circle detection process. Outliers in the dataset of circle centers can distort the alignment results, particularly when aligning images from different cameras like thermal and RGB, where consistent feature points are critical.
- **Method of Outlier Removal**:
This function operates by examining the dataset of detected circle centers and removing those that deviate significantly from their neighbors. The method evaluates the difference between consecutive points after sorting them, using a predefined maximum difference threshold (`max_diff`). Points that exhibit a difference greater than this threshold from their immediate neighbors are considered outliers and are excluded from the list of points used for further processing.

### 2.6 Homography Matrix Transformation
- **Principle of Perspective Transformation**:
A homography matrix transformation is a complex perspective transformation that uses a 3x3 matrix to map points from one image to corresponding positions in another image. This transformation can handle image rotation, scaling, translation, and more complex perspective changes.
- **Application of the Transformation**:
During calibration, the algorithm uses detected image feature points (such as circle centers) to calculate the homography matrix. The mapping relationships of these feature points guide the alignment of the images, ensuring that images captured from different cameras can be precisely overlaid.


## 3. Requirement
To run this calibration script, the following software and libraries are required:
### Step 1: Install Python
- **Python 3.8 or higher**: It is recommended to use Python 3.8 or newer to ensure compatibility with all dependencies. 
- You can download and install Python from [the official Python website](https://www.python.org/downloads/) or, for an easier management of packages and environments, use [Anaconda](https://www.anaconda.com/products/individual), which comes with most of the necessary scientific libraries pre-installed.
### Step 2: Set Up a Python Environment (Optional)
- **Creating by Conda**:
  If you are using Anaconda, setting up a virtual environment is highly recommended to avoid conflicts between project dependencies. You can create a conda environment with the following command:
  ```bash
  conda create --n calibration python=3.8
- **Using Conda**:
  After created the new conda environment, you can use the following command to activate the conda environment:
    ```bash
    conda activate calibration
- **Quitting Conda**:
    You can use the following command to quit the conda environment and go back to the base environment 
    ```bash
    conda deactivate
### Step 3: Install Required Packages
- **Install Necessary Libraries**: You can install the required libraries using pip. Run the following commands to install OpenCV and NumPy in you created calibration python environment:
  ```bash
  pip install numpy
  pip install opencv-python
## 4. Usage Guidelines
- **Configuration Files** Before running the script, ensure that the kd_intrinsics.txt and kdc_intrinsics.txt files are updated with the correct parameters. Despite the .txt extension, these files contain JSON-formatted data. 
- **Image Directory**: Before running the script, ensure that the images to be calibrated are correctly placed in the `SensorCommunication/Acquisition/calib_data` directory on your system. This directory should contain subfolders named according to the specific dataset they represent, such as `test_plant_20240412161903`.
- **Executing the Script**: 
    To run the calibration script, execute the following command in your terminal:

    ```bash
    python calibration.py
After running the command, you will be prompted to enter the name of the folder containing the images you wish to calibrate (e.g., test_plant_20240412161903). Enter the folder name, and the script will locate and process the images from the specified folder.

The calibrated images will be saved in the same directory (SensorCommunication/Acquisition/calib_data) under the original folder name. This ensures that you can easily find and distinguish between pre-calibrated and post-calibrated image sets.

