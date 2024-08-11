# Guide to Setting Up and Running the Camera Calibration Code
This guide provides step-by-step instructions for running the camera calibration code. These instructions are meant for users who have access to a terminal and are familiar with basic command line operations.

## 1. Purpose of the Calibration Code

This calibration code is designed to fine-tune and align images captured from both RGB and thermal cameras. The main objective is to ensure that these images are perfectly aligned, enhancing the accuracy and effectiveness of further image processing tasks like segmentation and analysis in plant phenotyping. 


## 2. Requirement
To run this calibration script, the following software and libraries are required:
### Step 1: Install Python
- **Python 3.8 or higher**: It is recommended to use Python 3.8 or newer to ensure compatibility with all dependencies. 
- You can download and install Python from [the official Python website](https://www.python.org/downloads/) or, for an easier management of packages and environments, use [Anaconda](https://www.anaconda.com/products/individual), which comes with most of the necessary scientific libraries pre-installed.
### Step 2: Set Up a Virtual Environment (Optional)
- **Using Conda**:
  If you are using Anaconda, setting up a virtual environment is highly recommended to avoid conflicts between project dependencies. You can create a conda environment with the following command:
  ```bash
  conda create --n calibration python=3.8
  conda activate calibration
### Step 3: Install Required Packages
- **Install Necessary Libraries**: You can install the required libraries using pip. Run the following commands to install OpenCV and NumPy:
  ```bash
  pip install numpy
  pip install opencv-python
## 3. Usage Guidelines
- **Configuration Files** Before running the script, ensure that the kd_intrinsics.txt and kdc_intrinsics.txt files are updated with the correct parameters. Despite the .txt extension, these files contain JSON-formatted data. 
- **Image Directory**: Before running the script, ensure that the images to be calibrated are correctly placed in the `SensorCommunication/Acquisition/calib_data` directory on your system. This directory should contain subfolders named according to the specific dataset they represent, such as `test_plant_20240412161903`.
- **Executing the Script**: 
    To run the calibration script, execute the following command in your terminal:

    ```bash
    python calibration.py
After running the command, you will be prompted to enter the name of the folder containing the images you wish to calibrate (e.g., test_plant_20240412161903). Enter the folder name, and the script will locate and process the images from the specified folder.

The calibrated images will be saved in the same directory (SensorCommunication/Acquisition/calib_data) under the original folder name. This ensures that you can easily find and distinguish between pre-calibrated and post-calibrated image sets.

