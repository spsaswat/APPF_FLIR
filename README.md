Based on the style you have provided and the detailed content from both the style and the current README, here is a sophisticated and comprehensive README for your project:

---

# APPF Overview

The Australian Plant Phenomics Facility (APPF) is a world-leading infrastructure facility that underpins innovative plant phenomics research to accelerate the development of new and improved crops, healthier food, and more sustainable agricultural practices. We operate nodes at the University of Adelaide and the Australian National University.

# APPF_FLIR Project Overview

This project aims to develop an advanced phenotyping system using a customized 3D hyperspectral and thermal scanner that can accurately identify the expression of food-grade recombinant proteins within plant leaves and perform thermal analysis of leaves. This system is designed to streamline phenotypic data collection and analysis, ultimately enhancing the breeding and cultivation of plants with desired traits.

## FLIR A645 SC Camera Overview

The FLIR A645 SC camera is utilized for thermal imaging, offering high-resolution long-wave infrared (LWIR) imaging, making it an ideal tool for scientific and research applications. Key features include:

- **Resolution:** 640 × 480 pixels for detailed imaging.
- **Thermal Sensitivity:** <50 mK at +30°C for accurate temperature measurement.
- **Temperature Range:** -20° to +650°C for diverse applications.
- **Connectivity:** 25 Hz gigabit ethernet, compatible with GiGE Vision and GenICam.
- **Lens:** Removable 24.5mm lens with a 25° x 18.8° field of view; interior threads for filters.

For more information, refer to the [FLIR A655sc Camera Specifications](https://www.flir.com.au/products/a655sc/).

### Applications

The FLIR A645 SC is perfect for applications requiring detailed thermal analysis, such as electronics inspection, battery evaluation, solar cell examination, and medical diagnostics.

## Realsense D405 Overview

The Intel® RealSense™ Depth Camera D405 provides sub-millimeter accuracy for close-range computer vision needs.

- **Resolution:** 1280 x 720 pixels
- **Range:** 0.07 to 0.5 meters; Operating Temperature: 0-55°C
- **Depth Range:** 7 cm to 50 cm
- **Depth Accuracy:** +/- 1.4% at 20 cm

For more information, please refer to the [Realsense D405 product page](https://www.framos.com/en/products/intel-realsense-depth-camera-d405-camera-only-26126).

### Applications

Ideal for defect inspection, pick & place operations, smart agriculture, and medical technology.

**In this project, the FLIR A645 SC camera is used to generate the thermal image of the leaf.**

## Testing Software

We employ the FLIR Thermal Studio Suite to receive data generated by the FLIR thermal camera. This software acts as the testing platform for the developed algorithms. It can be downloaded [here](https://www.flir.com.au/support/products/flir-thermal-studio-suite/#Downloads).

Further information can be found in [`testing.doc`](https://anu365.sharepoint.com/sites/APPF-TL-FLIR/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FAPPF%2DTL%2DFLIR%2FShared%20Documents%2FTesting&viewid=b4067fee%2D839d%2D4643%2D9b23%2D66e61f62ac63) within the landing page.

## Code Files Overview

### `camera_calibration.py`

This script calibrates the FLIR camera, computing intrinsic parameters necessary for accurate thermal image processing. It outputs calibration data for subsequent use.

### `capture_thermal_image.py`

This script captures thermal images using the FLIR camera, interfacing through the PySpin library, and saves the images to the designated directory.

### `thermal_image_processing.py`

Processes captured thermal images to correct distortions and apply necessary transformations using OpenCV. It relies on calibration data for precision.

### `thermal_image_analysis.py`

Analyzes processed thermal images to extract temperature data and other relevant features for further analysis. Results are saved for subsequent use.

### `Colormap_matplotlib.py`

Applies a colormap to thermal images for enhanced visualization using Matplotlib, facilitating better interpretation of thermal data.

## Setup Instructions

### 1. Activating the Conda Environment

Ensure you are in the project directory and activate the Conda environment to use the correct dependencies and Python version:

```bash
conda activate appf-flir
```

### 2. Camera Calibration

Run the calibration script to obtain necessary camera parameters:

```bash
python camera_calibration.py
```

### 3. Capturing Thermal Images

Capture images with the following command:

```bash
python capture_thermal_image.py
```

### 4. Processing Thermal Images

Process images using:

```bash
python thermal_image_processing.py
```

### 5. Analyzing Thermal Images

Analyze the images with:

```bash
python thermal_image_analysis.py
```

### 6. Visualizing Results with Colormap

Visualize processed images using:

```bash
python Colormap_matplotlib.py
```

### 7. Launching JupyterLab (Optional)

Launch JupyterLab for interactive exploration:

```bash
jupyter lab
```

## Relevant Links

- [Full SDK](https://flir.netx.net/file/asset/59416/original/attachment)
- [Python SDK](https://flir.netx.net/file/asset/59493/original/attachment)
- [Spinnaker FLIR SDK full and python](https://www.flir.com/support-center/iis/machine-vision/downloads/spinnaker-sdk-download/spinnaker-sdk--download-files/)
- [Project Landing Page](https://anu365.sharepoint.com/sites/APPF-TL-FLIR)
- [FLIR Corporation](https://Flir.com.au)

## Contact Information

For further inquiries or support, please contact the project team:

- **Saswat Panda**: saswat.panda@anu.edu.au
- **Ming-Dao Chia**: ming-dao.chia@anu.edu.au
- **Connor Li**: kexiao.li@anu.edu.au
- **Srikanth Polisetty**: srikanth.polisetty@anu.edu.au
- **Wanqi Qiu**: wanqi.qiu@anu.edu.au
- **Wenhao Wang**: wenhao.wang@anu.edu.au
- **Yongsong Xiao**: yongsong.xiao@anu.edu.au
- **Zehua Liu**: zehua.liu@anu.edu.au
- **Zishuang Xing**: zishuang.xing@anu.edu.au

---

This README combines the sophisticated style you prefer with the detailed content of the current README, providing a professional and comprehensive document for your project.
