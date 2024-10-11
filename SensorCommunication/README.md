# Project Title: Thermal Imaging Processing and Analysis with FLIR Cameras

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Setup Instructions](#setup-instructions)
   - [1. Activating the Conda Environment](#1-activating-the-conda-environment)
   - [2. Camera Calibration](#2-camera-calibration)
   - [3. Capturing Thermal Images](#3-capturing-thermal-images)
   - [4. Processing Thermal Images](#4-processing-thermal-images)
   - [5. Analyzing Thermal Images](#5-analyzing-thermal-images)
   - [6. Visualizing Results with Colormap](#6-visualizing-results-with-colormap)
   - [7. Launching JupyterLab (Optional)](#7-launching-jupyterlab-optional)
5. [Project Structure](#project-structure)
6. [Dependencies](#dependencies)
7. [Usage Notes](#usage-notes)
8. [Contributing](#contributing)
9. [License](#license)
10. [Contact Information](#contact-information)

---

## Introduction

This project provides a comprehensive framework for capturing, processing, analyzing, and visualizing thermal images using FLIR cameras. It includes scripts for camera calibration, image acquisition, image processing, data analysis, and result visualization, facilitating advanced thermal imaging applications.

## Prerequisites

- **Operating System**: Windows/Linux/macOS
- **Hardware**: FLIR thermal camera properly connected to your computer
- **Software**: Conda package manager

## Installation

Ensure that you have Conda installed on your system. If not, download and install Anaconda or Miniconda from the official website:

- [Anaconda Distribution](https://www.anaconda.com/products/distribution)
- [Miniconda Distribution](https://docs.conda.io/en/latest/miniconda.html)

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/APPF_FLIR.git
```

Navigate to the project directory:

```bash
cd APPF_FLIR
```

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate appf-flir
```

*Note: Ensure that the `environment.yml` file is present in the project directory, specifying all necessary dependencies.*

---

## Setup Instructions

### 1. Activating the Conda Environment

To begin working with the project, activate the Conda environment to ensure all dependencies and the correct Python version are used:

```bash
conda activate appf-flir
```

Verify that you are in the project directory:

```bash
D:\techlauncher\APPF_FLIR
```

The `appf-flir` environment should now be active.

### 2. Camera Calibration

Calibrate the FLIR camera to obtain intrinsic parameters necessary for accurate image processing. Run the calibration script:

```bash
python camera_calibration.py
```

This script computes the camera's intrinsic parameters and saves them to a file for future use.

### 3. Capturing Thermal Images

Capture thermal images using the FLIR camera by executing:

```bash
python capture_thermal_image.py
```

This script interfaces with the camera via the PySpin library and saves the captured images to the designated directory.

### 4. Processing Thermal Images

Process the captured images to correct distortions and apply necessary transformations:

```bash
python thermal_image_processing.py
```

The script utilizes OpenCV to undistort and scale the images based on the calibration data.

### 5. Analyzing Thermal Images

Extract temperature data and other relevant features from the processed thermal images:

```bash
python thermal_image_analysis.py
```

This script analyzes the images and saves the extracted data for further analysis.

### 6. Visualizing Results with Colormap

Visualize the thermal images using a colormap for enhanced interpretation:

```bash
python Colormap_matplotlib.py
```

The script applies a colormap using Matplotlib and saves the visualized images to the output directory.

### 7. Launching JupyterLab (Optional)

For interactive data exploration and analysis, launch JupyterLab:

```bash
jupyter lab
```

This command starts JupyterLab in your default web browser, providing an interface to work with notebooks and scripts. If the browser does not open automatically, navigate to the URL provided in the terminal.

---

## Project Structure

```
APPF_FLIR/
── data/
│   ├── raw/
│   ├── processed/
│   ── results/
── scripts/
│   ├── camera_calibration.py
│   ├── capture_thermal_image.py
│   ├── thermal_image_processing.py
│   ├── thermal_image_analysis.py
│   ── Colormap_matplotlib.py
── notebooks/
── environment.yml
── README.md
```

- **data/**: Contains raw, processed, and result data.
- **scripts/**: Includes all Python scripts for various stages of the workflow.
- **notebooks/**: Jupyter notebooks for interactive work.
- **environment.yml**: Conda environment specification file.
- **README.md**: Project documentation.

---

## Dependencies

The project relies on the following key packages:

- **Python**: 3.x
- **PySpin**: For interfacing with FLIR cameras.
- **OpenCV**: For image processing tasks.
- **NumPy**: For numerical computations.
- **Matplotlib**: For visualization.

All dependencies are specified in the `environment.yml` file. To install them, use:

```bash
conda env create -f environment.yml
```

---

## Usage Notes

- **Camera Connection**: Ensure the FLIR camera is properly connected before running any capture scripts.
- **Script Configuration**: Modify script parameters as needed to suit your specific setup (e.g., file paths, camera settings).
- **Data Management**: Organize captured and processed data within the `data/` directory structure for consistency.
- **Error Handling**: If you encounter issues, consult the script comments and ensure all dependencies are correctly installed.

---

## Contributing

Contributions to enhance the functionality or fix issues are welcome. Please follow these steps:

1. Fork the repository.
2. Create a new feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes with descriptive messages:
   ```bash
   git commit -m "Add feature: description"
   ```
4. Push to your forked repository:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a Pull Request detailing your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact Information

For questions or support, please contact:

- **Project Maintainers**: 
    - Saswat Panda (saswat.panda@anu.edu.au)
    - Ming-Dao Chia (ming-dao.chia@anu.edu.au)
    - Connor Li (kexiao.li@anu.edu.au)
    - Srikanth Polisetty (srikanth.polisetty@anu.edu.au)
    - Wanqi Qiu (wanqi.qiu@anu.edu.au)
    - Wenhao Wang (wenhao.wang@anu.edu.au)
    - Yongsong Xiao (yongsong.xiao@anu.edu.au) 
    - Zehua Liu (zehua.liu@anu.edu.au)
    - Zishuang Xing (zishuang.xing@anu.edu.au)

---

By following these instructions, you should be able to set up and utilize the thermal imaging processing and analysis tools effectively. Should you require further assistance, do not hesitate to reach out.