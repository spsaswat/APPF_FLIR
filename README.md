# APPF Overview

The Australian Plant Phenomics Facility (APPF) is a world-leading infrastructure facility that underpins innovative plant phenomics research to accelerate the development of new and improved crops, healthier food and more sustainable agricultural practice. We currently have nodes at the University of Adelaide and the Australian National University.


# APPF_FLIR Project Overview

This project aims to develop an advanced phenotyping system using a customized 3D hyperspectral scanner that can accurately identify the expression of food-grade recombinant proteins within plant leaves. This system is intended to streamline the process of phenotypic data collection and analysis, ultimately enhancing the breeding and cultivation of plants with desired traits.


## FLIR A645 SC Camera Overview

We are using FLIR A645 SC camera for thermal imaging. The FLIR A645 SC camera offers high-resolution long-wave infrared (LWIR) imaging, making it an ideal tool for scientific and research applications. Key features include:

- **Resolution:** 640 × 480 pixels, providing over 300,000 temperature measurement points.
- **Thermal Sensitivity:** <50 mK at +30°C (+86°F), ensuring precise temperature differentiation.
- **Temperature Range:** -20° to +650°C, suitable for a wide range of applications.
- **Connectivity:** 25 Hz gigabit ethernet with GiGE Vision and GenICam compatibility, facilitating integration with third-party analysis software.
- **Lens:** Removable 24.5mm lens with a field of view of 25° x 18.8°, complemented by interior threads for optional filter holders.

### Applications

The FLIR A645 SC is perfect for applications requiring detailed thermal analysis, such as electronics inspection, battery evaluation, solar cell examination, and medical diagnostics.

## Realsense D405 Overview

The Intel® RealSense™ Depth Camera D405 is a short-range stereo camera providing sub-millimeter accuracy for your close-range computer vision needs.

- **Manufacturer** Intel
- **Product Type** Housed Camera
- **Depth Technology:** Passive Stereo
- **Horizontal Resolution [px]:** 1280
- **Vertical Resolution [px]:** 720
- **Shutter Type:** Global Shutter
- **Field of View Horizontal [°]** 84
- **Field of View Vertical [°]** 58
- **Field of View Diagonal [°]** 92
- **Max. Depth Resolution** 1279 x 720
- **Max. Depth Frame Rate [fps]** 30
- **RGB Sensor** Yes
- **Max. RGB Resolution** 1280 x 720
- **Max. RGB Frame Rate [fps]** 30
- **Interface** USB3.1
- **Baseline [mm]** 18
- **Minimum Depth Distance [mm]** 100
- **Min. Operating Temperature (Backside Housing) [°C]** 0
- **Max. Operating Temperature (Backside Housing) [°C]** 55
- **Min. Storage Temperature (Ambient, Sustained) [°C]** 0
- **Max. Storage Temperature (Ambient, Sustained) [°C]** 50
- **Dimensions LxBxH [mm]** 42 x 42 x 23
- **Operating Relative Humidity [%]** 90
- **Ideal Range [m]** 0.07 to 0.5
- **Depth Filter** IR-Cut
- **Interial Measurement Unit (IMU)** No
- **Software Compatible** Intel RealSense SDK 2.0
- **Includes** Intel® RealSense™ D405 Depth Camera

### Applications 

Defect Inspection, Pick & Place, Smart Agriculture, Medical Technology

**In this project, we are using FLIR A645 SC camera to generate the thermal image of the leaf.**

## Testing Software
we are using FLIR Thermal Studio Suite to receive data generated by the FLIR thermal camera. In the circumstance of this project, it is considered to be the testing software for the developed algorithm. The software could be download [here](https://www.flir.com.au/support/products/flir-thermal-studio-suite/#Downloads).

More information is given in [`testing.doc`](https://anu365.sharepoint.com/sites/APPF-TL-FLIR/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FAPPF%2DTL%2DFLIR%2FShared%20Documents%2FTesting&viewid=b4067fee%2D839d%2D4643%2D9b23%2D66e61f62ac63) within the landing page.

## Relevant Links
[Spinnaker FLIR SDK full and python](https://www.flir.com/support-center/iis/machine-vision/downloads/spinnaker-sdk-download/spinnaker-sdk--download-files/)

[Landing Page of the Project](https://anu365.sharepoint.com/sites/APPF-TL-FLIR)

[FLIR Corporation](https://Flir.com.au)

[FLIR A655sc Camera Specifications](https://www.flir.com.au/products/a655sc/)

[Realsense D405](https://www.framos.com/en/products/intel-realsense-depth-camera-d405-camera-only-26126)
